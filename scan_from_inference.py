# USAGE:
# python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
# For example, to scan a single image with interactive mode:
# python scan.py --image sample_images/desk.JPG -i
# To scan all images in a directory automatically:
# python scan.py --images sample_images

# Scanned images will be output to directory named 'output'
import re
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
        # cv2.imshow("Dilated", dilated)
        # cv2.waitKey()

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
                if len(approx) == 4 and min_peri < peri < max_peri:
                    cnts_grid.append(c)
                    peris.append(peri)

        return cnts_grid, areas, peris

    def find_aoi(self, scanned, blur_kernel_size=5, debug=False):
        blur = cv2.GaussianBlur(scanned, (blur_kernel_size, blur_kernel_size), 0)
        if debug:
            cv2.imshow("Blur", blur)
            cv2.waitKey()

        thresh2 = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        if debug:
            cv2.imshow("thresh2", thresh2)
            cv2.waitKey()

        grid_contours, areas, peris = self.get_grid_contour(thresh2)

        if debug:
            cv2.drawContours(scanned, grid_contours, -1, (0, 0, 255), 1)
            cv2.imshow("Contours", scanned)
            cv2.waitKey()
            cv2.destroyAllWindows()

        #CREATING AOI FROM GRID CONTOURS
        try:
            concat = np.concatenate(grid_contours)
            hull = cv2.convexHull(concat)
            x, y, w, h = cv2.boundingRect(concat)

            return (x, y, w, h), hull, grid_contours, areas, peris

        except ValueError as e:
            print(e)
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="Directory of images to be scanned")
    ap.add_argument("--preds", help="Path to directory containing text predictions")
    ap.add_argument("-d", help='debug mode')
    args = vars(ap.parse_args())
    im_dir = Path(args["images"]) if args["images"] else None
    if not im_dir.is_dir():
        raise NotADirectoryError(f'Images directory not found: {args["images"]}')
    pred_dir = Path(args["preds"]) if args["preds"] else None
    if not pred_dir.is_dir():
        raise NotADirectoryError(f'Predictions directory not found: {args["images"]}')
    debug = True if args["d"] and args["d"] == 'True' else False

    scanner = DocScanner()

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    pred_globbed = pred_dir.glob(f"*out.txt")

    inf_pairs = []
    for pred in tqdm(pred_globbed):
        im_glob = list(im_dir.glob(f'**/{pred.stem}.png'))
        if len(im_glob) > 1:
            raise ValueError(f'There should only be one image for each inference. Got {len(im_glob)}\n'
                             f'Prediction: {pred}\n'
                             f'Images found: {im_glob}\n')
        elif len(im_glob) == 0:
            warnings.warn(f'No image found to match prediction:\n'
                          f'{pred}')
        else:
            im = im_glob[0]
            inf_pairs.append((im, pred))

    print(f'Found {len(inf_pairs)} pairs of images and predictions. Detecting numbers in predicted bboxes...')

    for img, prediction in inf_pairs:
        inf_data = pandas.read_table(prediction, sep=' ', header=None)
        scanned_color = cv2.imread(str(img))
        scanned = cv2.cvtColor(scanned_color, cv2.COLOR_BGR2GRAY)
        if not scanned.shape[1] == 1024:
            warnings.warn(f'Images used for inferences should already be preprocessed and resized to a 1024 height\n'
                          f'Got image of shape: {scanned.shape}')
            ratio = 1024 / scanned.shape[0]
            scanned_color = cv2.resize(scanned_color, (round(scanned_color.shape[1] * ratio), 1024), interpolation=cv2.INTER_LINEAR)
            scanned = cv2.resize(scanned, (round(scanned.shape[1]*ratio), 1024), interpolation=cv2.INTER_NEAREST)

        # find grid
        bbox, hull, grid_contours, areas, peris = scanner.find_aoi(scanned)  # bbox: x, y, w, h

        # BLACK OUT CHECKED NUMBERS IN IMAGES
        scanned_draw = scanned.copy()
        #mask = np.zeros((scanned.shape), np.uint8)
        conf_threshold = 0.77  # FIXME: softcode
        for row in inf_data.itertuples():
            conf = row[-2]
            if conf > conf_threshold:
                x_rel, y_rel, w_rel, h_rel = row[-6:-2]
                pt1 = int((x_rel-w_rel/2)*scanned.shape[1]), int((y_rel-h_rel/2)*scanned.shape[0])
                pt2 = int((x_rel+w_rel/2)*scanned.shape[1]), int((y_rel+h_rel/2)*scanned.shape[0])
                cv2.rectangle(scanned_draw, pt1, pt2, 0, -1)

        if debug:
            cv2.imshow("Non-checked", scanned_draw)
            cv2.waitKey()
            cv2.destroyAllWindows()

        continue

        # Output numbers of checked boxes

        area_hull = cv2.contourArea(hull)
        peri = cv2.arcLength(hull, True)

        area_bb = ((bbox[0]+bbox[2])-bbox[0])*((bbox[1]+bbox[3])-bbox[1])
        hull_bb_ratio = area_hull/area_bb*100

        # DRAWING RECTANGLE ON AOI
        mask = np.zeros((scanned.shape), np.uint8)
        # draw white rectangle on aoi
        image = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        # add black outline of thickness on non-aoi TODO: necessary to add this outline?
        image = cv2.rectangle(mask, (x, y), (x + w, y + h), 0, 2)

        out_ckd = np.zeros_like(scanned)
        mask_ckd = mask.copy()
        # BLACK OUT WHERE CONTOURS (non-checked grid items) ARE ON SCANNED IMAGE
        for i, contour in enumerate(grid_contours):
            # cont_hull = cv2.convexHull(contour)
            xb, yb, wb, hb = cv2.boundingRect(contour)
            # black out where contours are on scanned image
            image = cv2.rectangle(mask_ckd, (xb, yb), (xb + wb, yb + hb), 0, -1)
            # crop individual contour for saving
            grid_item = scanned[yb:yb + hb, xb:xb + wb]

        # where mask is white, restore original values (i.e. aoi values)
        out_ckd[mask_ckd == 255] = scanned[mask_ckd == 255]  # only use if using previous for loop

        # on out non checked, we don't restore original values, we only restore non checked grid items
        out_non_ckd = np.zeros_like(scanned)
        mask_non_ckd = mask.copy()

        # IN AOI, KEEP EVERYTHING BLACK EXCEPT CONTOURS (non-checked grid items)
        for i, contour in enumerate(grid_contours):
            # cont_hull = cv2.convexHull(contour)
            xb, yb, wb, hb = cv2.boundingRect(contour)
            out_non_ckd[yb:yb + hb, xb:xb + wb] = scanned[yb:yb + hb, xb:xb + wb]

        out_non_ckd_draw = out_non_ckd.copy()
        # WRITING ONLY AOI AS IMAGE
        outdir_trn = Path(f'/home/remi/Documents/sherbrooke_citoyen/training_img_15oct/{im.parent.parent.parent.name}')
        Path.mkdir(outdir_trn, exist_ok=True)
        outfile = outdir_trn / f'{im.stem}_trn.png'
        outfile_draw = outdir_trn / f'{im.stem}_bbox.png'
        # outfile = outdir_trn / f'{im.stem}_aoi.png'
        # cv2.imwrite(str(outfile), out_ckd)

        if debug:
            cv2.imshow("New image", out_ckd)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # FROM HERE, WE FILTER GRID ITEMS THAT ARE CHECKED FOR SURE AND ADD THEM TO OUT_NON_CKD
        # THIS WILL SERVE AS TRAINING DATA
        blur2 = cv2.GaussianBlur(out_ckd, (5, 5), 0)
        if debug:
            cv2.imshow("Blur2", blur2)
            cv2.waitKey()

        # DILATE WITH WHITE TO TRY AND GET INDIVIDUAL CHECKED GRID ITEMS
        # dilate helps to separate checked "x" from sides of grid item
        MORPH = 5
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
        dilated2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel2)
        if debug:
            cv2.imshow("Dilated2", dilated2)
            cv2.waitKey()

        # FIRST PASS TO GET INDIVIDUAL CHECKED GRID ITEMS
        grid_contours2, areas, peris = scanner.get_grid_contour(dilated2)

        trn_bbox = []
        # ADD ITEMS WE KNOW FOR SURE ARE CHECKED (FILTER BY AREA, THEN SHAPE, THEN MEAN VALUE OR
        # LARGEST CONTOUR IN SINGLE GRID ITEM --> SEE FILTER_CHECKED() FUNCTION FOR DETAILS)
        for i, contour in enumerate(grid_contours2):
            area = cv2.contourArea(contour)
            areas.append(area)
            # approx area of a single grid item
            if 1500 < area < 2500:
                ctn = DocScanner.filter_by_shape(contour)
                if ctn is not None:
                    xc, yc, wc, hc = cv2.boundingRect(contour)
                    grid_item = scanned[yc:yc + hc, xc:xc + wc]
                    grid_item = DocScanner.filter_checked(grid_item)
                    if grid_item is not None:
                        out_non_ckd[yc:yc + hc, xc:xc + wc] = scanned[yc:yc + hc, xc:xc + wc]
                        out_non_ckd_draw[yc:yc + hc, xc:xc + wc] = scanned[yc:yc + hc, xc:xc + wc]
                        cv2.rectangle(out_non_ckd_draw, (xc, yc), (xc + wc, yc + hc), 122, 2)
                        trn_bbox.append((xc, yc, wc, hc))

        # SECOND PASS TO GET CHECKED GRID ITEMS THAT ARE STUCK TOGETHER IN PAIRS (NO MORE THAN PAIR)
        # MOST HORIZONTAL AND VERTICAL PAIRS ARE EXTRACTED AND KEPT FOR TRAINING
        grid_contours3, areas, peris = scanner.get_grid_contour(dilated2, min_area=3000, max_area=5000,
                                                                min_peri=200, max_peri=350)
        print(im.stem)
        print(len(grid_contours3))
        for i, contour in enumerate(grid_contours3):
            area = cv2.contourArea(contour)
            areas.append(area)
            xc, yc, wc, hc = cv2.boundingRect(contour)
            grid_item_uncut = scanned[yc:yc + hc, xc:xc + wc]
            if debug:
                cv2.imshow(f"{im.stem}_{area}", grid_item_uncut)
                cv2.waitKey()
            if 3000 < area < 5000:
                ctn = DocScanner.filter_by_shape(contour, min_peri=210, max_peri=350)
                if ctn is None:
                    continue
                # FILLING OUTSIDE OF CONTOUR WITH BLACK
                mask2 = np.zeros((scanned.shape), np.uint8)
                cv2.drawContours(mask2, [contour], 0, 255, -1)
                out2 = np.zeros_like(scanned)
                out2[mask2 == 255] = scanned[mask2 == 255]
                if debug:
                    cv2.imshow(f"out2_{area}", out2)
                    cv2.waitKey()
                # crop individual contour for saving
                if wc > hc:
                    for x in range(xc, xc+wc, int(wc/2)):
                        # crop individual grid item
                        grid_item = out2[yc:yc + hc, x:x + int(wc/2)]
                        if debug:
                            cv2.imshow(f"grid item cut", grid_item)
                            cv2.waitKey()
                        grid_item = DocScanner.filter_checked(grid_item)
                        if grid_item is not None:
                            out_non_ckd[yc:yc + hc, x:x + int(wc/2)] = scanned[yc:yc + hc, x:x + int(wc/2)]
                            bbox = (x, yc, int(wc/2), hc)
                            out_non_ckd_draw[yc:yc + hc, x:x + int(wc/2)] = scanned[yc:yc + hc, x:x + int(wc/2)]
                            cv2.rectangle(out_non_ckd_draw, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 122, 2)
                            trn_bbox.append(bbox)
                else:
                    for y in range(yc, yc+hc, int(hc/2)):
                        grid_item = out2[y:y + int(hc/2), xc:xc + wc]
                        if debug:
                            cv2.imshow(f"grid item cut", grid_item)
                            cv2.waitKey()
                        grid_item = DocScanner.filter_checked(grid_item)
                        if grid_item is not None:
                            out_non_ckd[y:y + int(hc/2), xc:xc + wc] = scanned[y:y + int(hc/2), xc:xc + wc]
                            bbox = (xc, y, wc, int(hc/2))
                            out_non_ckd_draw[y:y + int(hc / 2), xc:xc + wc] = scanned[y:y + int(hc / 2), xc:xc + wc]
                            cv2.rectangle(out_non_ckd_draw, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 122, 2)
                            trn_bbox.append(bbox)


        with open(outdir_trn.parent / 'bingo_trn.csv', 'a') as fh:
            line = f'{outfile};{trn_bbox}\n'
            fh.write(line)

        # Now that filtering of checked items is complete, restore original value where mask is black (non-aoi)
        out_non_ckd[mask == 0] = scanned[mask == 0]
        cv2.imwrite(str(outfile), out_non_ckd)
        cv2.imwrite(str(outfile_draw), out_non_ckd_draw)
