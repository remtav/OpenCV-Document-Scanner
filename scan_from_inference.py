# USAGE:
# python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
# For example, to scan a single image with interactive mode:
# python scan.py --image sample_images/desk.JPG -i
# To scan all images in a directory automatically:
# python scan.py --images sample_images

# Scanned images will be output to directory named 'output'
import glob
import re
import shutil
from io import BytesIO
from pathlib import Path

import pandas
from PIL import Image
from pytesseract import pytesseract
from tqdm import tqdm

from pyimagesearch import transform
from pyimagesearch import imutils
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import polygon_interacter as poly_i
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd
from pdf2image import convert_from_path

import argparse
import os
import warnings


class DocScanner(object):
    """An image scanner"""

    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        """
        Args:
            interactive (boolean): If True, user can adjust screen contour before
                transformation occurs in interactive pyplot window.
            MIN_QUAD_AREA_RATIO (float): A contour will be rejected if its corners
                do not form a quadrilateral that covers at least MIN_QUAD_AREA_RATIO
                of the original image. Defaults to 0.25.
            MAX_QUAD_ANGLE_RANGE (int):  A contour will also be rejected if the range
                of its interior angles exceeds MAX_QUAD_ANGLE_RANGE. Defaults to 40.
        """
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE

    @staticmethod
    def filter_by_shape(contour, min_peri=150, max_peri=230):
        # cnts_grid.append(c)
        # # approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # if the contour has four vertices, then we have found
        # rectangular contours
        if len(approx) == 4 and min_peri < peri < max_peri:
            peris.append(peri)

            # FILLING OUTSIDE OF CONTOUR WITH BLACK
            mask2 = np.zeros((scanned.shape), np.uint8)
            cv2.drawContours(mask2, [contour], 0, 255, -1)
            if debug:
                cv2.imshow("mask2", mask2)
                cv2.waitKey()
            out2 = np.zeros_like(scanned)
            out2[mask2 == 255] = scanned[mask2 == 255]
            if debug:
                cv2.imshow("out2", out2)
                cv2.waitKey()

            return contour
        else:
            return None

    @staticmethod
    def filter_checked(grid_item):
        if debug:
            cv2.imshow("Grid item", grid_item)
            cv2.waitKey()
        thresh_it = cv2.adaptiveThreshold(grid_item, 255, 1, 1, 11, 2)
        if debug:
            cv2.imshow("thresh item", thresh_it)
            cv2.waitKey()
        MORPH_it = 8
        kernel_it = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_it, MORPH_it))
        dilated_it = cv2.morphologyEx(thresh_it, cv2.MORPH_CLOSE, kernel_it)
        if debug:
            cv2.imshow("Dilated item", dilated_it)
            cv2.waitKey()
        mean_val = int(np.mean(grid_item))
        # under 185 is certainly checked
        if 120 < mean_val < 165:
            return grid_item
        elif 120 < mean_val:
            # bringing it back white insides, black outsides
            dilated_it = cv2.adaptiveThreshold(dilated_it, 255, 1, 1, 11, 2)
            last_filtered, _ = cv2.findContours(dilated_it, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            last_filtered = sorted(last_filtered, key=cv2.contourArea, reverse=True)
            largest = cv2.contourArea(last_filtered[0])
            cv2.drawContours(dilated_it, [last_filtered[0]], -1, 125, 2)
            if debug:
                cv2.imshow('Dilated_it_cont', dilated_it)
                cv2.waitKey()

            # IF LARGEST CONTOUR IS BIGGER THAN 1000, WE PROBABLY HAVE A NON-CHECKED GRID ITEM
            if 165 < mean_val < 194 and largest < 1000:
                return grid_item
            else:
                return None

    def get_grid_contour(self, rescaled_image, min_area=1500, max_area=2300, min_peri=150, max_peri=210):
        """
        Returns a numpy array of shape (4, 2) containing the vertices of the four corners
        of the document in the image. It considers the corners returned from get_corners()
        and uses heuristics to choose the four corners that most likely represent
        the corners of the document. If no corners were found, or the four corners represent
        a quadrilateral that is too small or convex, it returns the original four corners.
        """

        # these constants are carefully chosen
        MORPH = 5
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH = rescaled_image.shape

        # convert the image to grayscale and blur it slightly
        # gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (7,7), 0)
        gray = rescaled_image


        # dilate helps to remove potential holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        if debug:
            cv2.imshow("Dilated", dilated)
            cv2.waitKey()

        # find edges and mark them in the output map using the Canny algorithm
        # edged = cv2.Canny(dilated, 0, CANNY)
        # test_corners = self.get_corners(edged)
        # cv2.imshow("Edged", edged)
        # cv2.waitKey()
        edged = dilated

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cnts_grid = []
        areas = []
        peris = []
        discarded_peris = []
        discarded_approx = []
        for c in contours:
            area = cv2.contourArea(c)
            areas.append(area)
            if min_area < area < max_area:
                #cnts_grid.append(c)
                # # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if the contour has four vertices, then we have found
                # rectangular contours
                if len(approx) == 4:
                    if min_peri < peri < max_peri:
                        cnts_grid.append(c)
                        peris.append(peri)
                    else:
                        discarded_peris.append(peri)
                else:
                    discarded_approx.append(len(approx))

        return cnts_grid, areas, peris

    def find_aoi(self, scanned, blur_kernel_size=5,
                 min_area=1500, max_area=2300,
                 min_peri=150, max_peri=210,
                 debug=False):
        blur = cv2.GaussianBlur(scanned, (blur_kernel_size, blur_kernel_size), 0)
        if debug:
            cv2.imshow("Blur", blur)
            cv2.waitKey()

        thresh2 = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        if debug:
            cv2.imshow("thresh2", thresh2)
            cv2.waitKey()

        grid_contours, areas, peris = self.get_grid_contour(thresh2,
                                                            min_area=min_area,
                                                            max_area=max_area,
                                                            min_peri=min_peri,
                                                            max_peri=max_peri)

        if debug:
            scanned_ctns = scanned.copy()
            cv2.drawContours(scanned_ctns, grid_contours, -1, 122, 3)
            cv2.imshow("Contours", scanned_ctns)
            cv2.waitKey()

        #CREATING AOI FROM GRID CONTOURS
        try:
            concat = np.concatenate(grid_contours)
            hull = cv2.convexHull(concat)

            return concat, hull, grid_contours, areas, peris

        except ValueError as e:
            print(e)
            return

    def load_images(self, path):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if Path(x).suffix.lower() == '.png']
        return images


def magic_num_from_pt(x_pt, y_pt, x_box, y_box, w_box, h_box):
    # get relative position to width of bbs --> 25 columns
    columns_limits = np.arange(x_box, w_box, w_box / 25)
    column_num = np.max(np.where(columns_limits < x_pt))
    # get relative position to height of bbs --> 18 rows
    row_limits = np.arange(y_box, h_box, h_box / 18)
    row_num = np.max(np.where(row_limits < y_pt))
    magic_number = row_num * 25 + column_num + 1
    return magic_number, columns_limits, row_limits, column_num, row_num

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="Directory of images to be scanned")
    ap.add_argument("--preds", help="Path to directory containing text predictions")
    ap.add_argument("-d", action='store_true', help='debug mode')
    args = vars(ap.parse_args())
    im_dir = Path(args["images"]) if args["images"] else None
    if not im_dir.is_dir():
        warnings.warn(f'Images directory not found: {args["images"]}')
    pred_dir = Path(args["preds"]) if args["preds"] else None
    if not pred_dir.is_dir():
        raise NotADirectoryError(f'Predictions directory not found: {args["images"]}')
    debug = args["d"]

    scanner = DocScanner()

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    images = scanner.load_images(im_dir)

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    #pred_globbed = list(pred_dir.glob(f"**/*out.txt"))

    inf_pairs = []
    for img in tqdm(images):
        img_district = [dir for dir in img.split('/') if 'District' in dir][0]
        pred = list(pred_dir.glob(f"{img_district}/{Path(img).stem}.txt"))
        # for pred_file in pred_globbed:
        #     if pred_file.stem == Path(img).stem and pred_file.parts[-2] == img_district:
        #         pred = pred_file
        if len(pred) > 1:
            raise ValueError(f'There should only be one prediction for each image. Got {len(pred)}\n'
                             f'Image: {img}\n'
                             f'Predictions found: {pred}\n')
        elif len(pred) == 0:
            warnings.warn(f'No image found to match prediction:\n'
                          f'{pred}')
        else:
            pred = pred[0]
            inf_pairs.append((img, pred))

    print(f'Found {len(inf_pairs)} pairs of images and predictions. Detecting numbers in predicted bboxes...')

    last_district = None
    results = []

    failed = 0
    validate = 0
    success = 0
    for img, prediction in tqdm(inf_pairs):
        # WRITING ONLY AOI AS IMAGE
        img_district = [dir for dir in img.split('/') if 'District' in dir][0]
        outdir = Path(f'/home/remi/Documents/sherbrooke_citoyen/27oct_from_inf/{img_district}')
        Path.mkdir(outdir, exist_ok=True)
        outdir_2nd = outdir / 'to_validate'
        Path.mkdir(outdir_2nd, exist_ok=True)
        outdir_failed = outdir / 'failed'
        Path.mkdir(outdir_failed, exist_ok=True)
        prediction_img = prediction.parent / f'{prediction.stem}.png'
        outfile = outdir / f'{Path(img).stem}_aoi.png'
        # outfile_draw = outdir_trn / f'{im.stem}_bbox.png'
        try:
            inf_data = pandas.read_table(prediction, sep=' ', header=None)
            scanned_color = cv2.imread(str(img))
            scanned = cv2.cvtColor(scanned_color, cv2.COLOR_BGR2GRAY)
            if not scanned.shape[1] == 1024:
                # warnings.warn(f'Images used for inferences should already be preprocessed and resized to a 1024 height\n'
                #               f'Got image of shape: {scanned.shape}')
                ratio = 1024 / scanned.shape[0]
                scanned_color = cv2.resize(scanned_color, (round(scanned_color.shape[1] * ratio), 1024), interpolation=cv2.INTER_LINEAR)
                scanned = cv2.resize(scanned, (round(scanned.shape[1]*ratio), 1024), interpolation=cv2.INTER_NEAREST)

            # BLACK OUT CHECKED NUMBERS IN IMAGES
            scanned_draw = scanned.copy()
            mask_hull = np.zeros((scanned.shape), np.uint8)
            mask_2unwarp = np.zeros((scanned.shape), np.uint8)

            conf_threshold = 0.77  # FIXME: softcode
            pred_areas = []
            pred_peris = []
            pred_ratios = []
            pred_pts = []
            centers = []
            for row in inf_data.itertuples():
                conf = row[-2]
                if conf > conf_threshold:
                    x_rel, y_rel, w_rel, h_rel = row[-6:-2]
                    x = int(x_rel * scanned.shape[1])
                    y = int(y_rel * scanned.shape[0])
                    w = int(w_rel * scanned.shape[1])
                    h = int(h_rel * scanned.shape[0])
                    ratio = w / h
                    if 0.8 < ratio < 1.3:
                        pt1 = int((x_rel-w_rel/2)*scanned.shape[1]), int((y_rel-h_rel/2)*scanned.shape[0])
                        pt2 = int((x_rel+w_rel/2)*scanned.shape[1]), int((y_rel+h_rel/2)*scanned.shape[0])
                        pred_pts.append(pt1)
                        pred_pts.append(pt2)
                        center = (x, y)
                        centers.append(center)
                        cv2.rectangle(mask_hull, pt1, pt2, 255, -1)
                        cv2.circle(scanned_draw, center, 1, 122, thickness=4)
                        cv2.circle(mask_2unwarp, center, 1, 255, thickness=4)
                        area = w*h
                        peri = 2*w+2*h
                        pred_areas.append(area)
                        pred_peris.append(peri)
                        pred_ratios.append(ratio)

            if debug:
                cv2.imshow("Checked", scanned_draw)
                cv2.waitKey()
                cv2.destroyAllWindows()

            # find grid
            mean_area = np.mean(pred_areas)
            min_area = min(pred_areas)
            max_area = max(pred_areas)
            min_peri = min(pred_peris)
            max_peri = max(pred_peris)
            concat, hull, grid_contours, areas, peris = scanner.find_aoi(scanned,
                                                                         min_area=min_area*0.65,
                                                                         max_area=max_area*1.15,
                                                                         min_peri=min_peri*0.75,
                                                                         max_peri=max_peri*1.15,
                                                                         debug=debug)  # bbox: x, y, w, h

            # DRAWING HULL ON AOI
            # draw white on aoi (concatenate prediction points with grid contour)
            pred_pts_arr = np.array(pred_pts)
            concat = np.concatenate([concat[:, 0, :], pred_pts_arr])
            hull = cv2.convexHull(concat)
            mask_hull = cv2.drawContours(mask_hull, [hull], -1, 255, -1)
            # where mask is black, paint original image in black
            scanned_draw[mask_hull == 0] = 0

            if debug:
                cv2.imshow("AOI_mask", mask_hull)
                cv2.waitKey()

            # DRAWING BB ON AOI
            mask_bb = np.zeros((scanned.shape), np.uint8)
            x, y, w, h = cv2.boundingRect(concat)
            # DRAWING WHITE RECTANGLE ON AOI
            aoi_bb = cv2.rectangle(mask_bb, (x, y), (x + w, y + h), 255, -1)
            # where mask is black, paint original image in black
            #scanned_draw[mask_bb == 0] = 0

            component1 = mask_hull.astype(bool)
            component2 = mask_bb.astype(bool)

            overlap = component1 * component2  # Logical AND
            union = component1 + component2  # Logical OR

            # Treats "True" as 1, sums number of Trues in overlap and union and divides
            IOU = overlap.sum() / float(union.sum())

            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.005 * peri, True)
            mask_hull = cv2.drawContours(mask_hull, [approx], -1, 122, 6)

            if debug:
                cv2.imshow("Checked", scanned_draw)
                cv2.waitKey()
                cv2.imshow("AOI_mask", mask_hull)
                cv2.waitKey()
                cv2.imshow(f"AOI_mask BB {IOU:.2f}", aoi_bb)
                cv2.waitKey()
                cv2.destroyAllWindows()

            if IOU > 0.975:
                # loop through predicted bbs
                numbers = []
                for x_cent, y_cent in centers:
                    magic_number, *_ = magic_num_from_pt(x_cent, y_cent, x, y, w, h)
                    mask_pt = cv2.circle(scanned_draw, (x_cent, y_cent), 1, 0, thickness=12)
                    if not 0 < magic_number <= 450:
                        cv2.imshow(f"{magic_number}", scanned_draw)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                    numbers.append(round(magic_number))
                    if debug:
                        cv2.imshow(f"{magic_number}", scanned_draw)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                results.append(f'{img_district};good;{Path(img).name};{sorted(numbers)}\n')
                success += 1
                shutil.copy(prediction_img, outdir / prediction_img.name)
                cv2.imwrite(str(outdir / f'{prediction_img.stem}_grid.png'), mask_pt)
            # if the contour has four vertices, then we have found rectangular contours
            elif len(approx) == 4:
                if debug:
                    cv2.imshow(f"{len(hull)} {len(approx)}", mask_hull)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                # get approx corners ((x,y) tuples)
                points = approx[:, 0, :]  # np.array(approx)
                points = transform.order_points(points)
                #points = np.array([[p] for p in points], dtype="int32")
                # unwarp to rectangle
                unwarped = transform.four_point_transform(mask_2unwarp, points)
                unwarped_scanned = transform.four_point_transform(scanned_draw, points)
                # use previous technique to read numbers and voilà!
                if debug:
                    cv2.imshow(f"before", scanned_draw)
                    cv2.waitKey()
                    cv2.imshow(f"after", unwarped)
                    cv2.waitKey()
                warped_ctns, _ = cv2.findContours(unwarped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                numbers = []
                if len(warped_ctns) != len(centers):
                    raise ValueError
                unwarped_scanned_vis = unwarped_scanned.copy()
                for c in warped_ctns:
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    magic_number, col_lims, row_lims, *_ = magic_num_from_pt(cX, cY, 0, 0, unwarped.shape[1], unwarped.shape[0])
                    for col_lim in col_lims:
                        cv2.line(unwarped_scanned_vis, (int(col_lim), 0), (int(col_lim), unwarped_scanned_vis.shape[0]), 122, thickness=3)
                    for row_lim in row_lims:
                        cv2.line(unwarped_scanned_vis, (0, int(row_lim)), (unwarped_scanned_vis.shape[1], int(row_lim)), 122, thickness=3)
                    mask_pt = cv2.circle(unwarped_scanned_vis, (cX, cY), 1, 0, thickness=8)
                    numbers.append(magic_number)
                if debug:
                    cv2.imshow(f"{sorted(numbers)}", mask_pt)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                results.append(f'{img_district};moderate;{Path(img).name};{sorted(numbers)}\n')
                validate += 1
                shutil.copy(prediction_img, outdir_2nd / prediction_img.name)
                cv2.imwrite(str(outdir_2nd / f'{prediction_img.stem}_grid.png'), mask_pt)
            else:
                failed += 1
                #shutil.copy(prediction_img, outdir_failed / prediction_img.name)
        except (TypeError, ValueError) as e:
            failed += 1
            print(e)
            shutil.copy(prediction_img, outdir_failed / prediction_img.name)

        outdir_res = Path(f'/home/remi/Documents/sherbrooke_citoyen/27oct_from_inf')
        with open(outdir_res / 'results.csv', 'w') as fh:
            fh.writelines(results)

    print(f"Failed: {failed}\nTo validate: {validate}\nSuccess: {success}")
