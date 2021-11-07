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

    def filter_corners(self, corners, min_dist=20):
        """Filters corners that are within min_dist of others"""
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_between_vectors_degrees(self, u, v):
        """Returns the angle between two vectors in degrees"""
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        """
        Returns the angle between the line segment from p2 to p1 
        and the line segment from p2 to p3 in degrees
        """
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        """
        Returns the range between max and min interior angles of quadrilateral.
        The input quadrilateral must be a numpy array with vertices ordered clockwise
        starting with the top left vertex.
        """
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)          

    def get_corners(self, img):
        """
        Returns a list of corners ((x, y) tuples) found in the input image. With proper
        pre-processing and filtering, it should output at most 10 potential corners.
        This is a utility function used by get_contours. The input image is expected 
        to be rescaled and Canny filtered prior to be passed in.
        """
        lines = lsd(img)

        # massages the output from LSD
        # LSD operates on edges. One "line" has 2 edges, and so we need to combine the edges back into lines
        # 1. separate out the lines into horizontal and vertical lines.
        # 2. Draw the horizontal lines back onto a canvas, but slightly thicker and longer.
        # 3. Run connected-components on the new canvas
        # 4. Get the bounding box for each component, and the bounding box is final line.
        # 5. The ends of each line is a corner
        # 6. Repeat for vertical lines
        # 7. Draw all the final lines onto another canvas. Where the lines overlap are also corners

        corners = []
        if lines is not None:
            # separate out the horizontal and vertical lines, and draw them back onto separate canvases
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # find the corners
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # remove corners in close proximity
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """Returns True if the contour satisfies all requirements set at instantitation"""

        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)


    def get_contour(self, rescaled_image):
        """
        Returns a numpy array of shape (4, 2) containing the vertices of the four corners
        of the document in the image. It considers the corners returned from get_corners()
        and uses heuristics to choose the four corners that most likely represent
        the corners of the document. If no corners were found, or the four corners represent
        a quadrilateral that is too small or convex, it returns the original four corners.
        """

        # these constants are carefully chosen
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)


        # dilate helps to remove potential holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # find edges and mark them in the output map using the Canny algorithm
        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype="int32")
                quads.append(points)

            # get top five quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # sort candidate quadrilaterals by their angle range, which helps remove outliers
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            # for debugging: uncomment the code below to draw the corners and countour found 
            # by get_corners() and overlay it on the image

            # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*test_corners))
            # plt.imshow(rescaled_image)
            # plt.show()

        # also attempt to find contours directly from the edged image, which occasionally 
        # produces better results
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # If we did not find any valid contours, just use the whole image
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(('Drag the corners of the box to the corners of the document. \n'
            'Close the window when finished.'))
        p = poly_i.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype = "int32")
        return new_points.reshape(4, 2)

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

    def scan(self, image_path, debug=False):
        # print('TESTING. HARCODED STUFF.')
        # image = cv2.imread(str(image_path))
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # return gray
        # output info
        suffix = '.png'  # if image_path.suffix == '.pdf' else image_path.suffix
        outdir_scan = image_path.parent / "scanned"
        outdir_scan.mkdir(exist_ok=True)
        outfile = outdir_scan / f"{image_path.stem}_out{suffix}"
        if outfile.is_file():
            print(f'Exists:{outfile}')
            image = cv2.imread(str(outfile))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray

        RESCALED_HEIGHT = 500.0
        #OUTPUT_DIR = 'output'

        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = cv2.imread(str(image_path))

        assert(image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height = int(RESCALED_HEIGHT))

        # get the contour of the document
        screenCnt = self.get_contour(rescaled_image)

        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        # apply the perspective transformation on color images
        hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsvimg[..., 0])
        mean_sat = np.mean(hsvimg[..., 1])
        # filtering based on hue and saturation values. Not perfect, but works in most cases
        warp = True if mean_hue > 1 and mean_sat > 30 else False

        if warp:
            warped = transform.four_point_transform(orig, screenCnt * ratio)
        else:
            warped = orig

        # convert the warped image to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # sharpen image
        sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
        
        # apply adaptive threshold to get black and white effect
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        if debug:
             cv2.imshow("thresh", thresh)
             cv2.waitKey()

        print(thresh.shape)

        osd = pytesseract.image_to_osd(thresh)
        angle = int(re.search('(?<=Rotate: )\d+', osd).group(0))
        print("angle: ", angle)
        #script = re.search('(?<=Script: )\d+', osd).group(0)
        #print("script: ", script)
        if debug:
            cv2.imshow(f'angle: {angle}', gray)
            cv2.waitKey()
            cv2.destroyAllWindows()

        if angle % 90 != 0:
            raise ValueError(f'Angle {angle} should be a multiple of 90')
        num_rots = int((360-angle) / 90)

        thresh = np.rot90(thresh, num_rots)

        # save the transformed image
        cv2.imwrite(str(outfile), thresh)
        print(f"Processed {outfile}")

        return thresh

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

        # contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #
        # # loop over the contours
        # cnts_rec = []
        # for c in contours:
        #     # approximate the contour
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        #     # if the contour has four vertices, then we have found
        #     # rectangular contours
        #     if len(approx) == 4:
        #         cnts_rec.append(c)
        # #
        # # mask = np.zeros((thresh2.shape), np.uint8)
        # #cv2.drawContours(rescaled_image, contours, 0, 255, -1)
        # cv2.drawContours(rescaled_image, cnts_rec, -1, 120, 2)
        # if debug:
        #      cv2.imshow("Contours", rescaled_image)
        #      cv2.waitKey()

        approx_contours = []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true',
        help = "Flag for manually verifying and/or setting document corners")
    ap.add_argument("-d", help='debug mode')
    args = vars(ap.parse_args())
    im_dir = args["images"]
    im_file_path = args["image"]
    debug = True if args["d"] and args["d"] == 'True' else False
    interactive_mode = args["i"]

    scanner = DocScanner(interactive_mode)

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        im_file_path = Path(im_file_path)
        if im_file_path.suffix == '.pdf':
            raise NotImplementedError('Use --images arg with pdf files')
        scanner.scan(im_file_path)

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        im_files = []
        for ext in valid_formats:
            #im_files.extend(Path(im_dir).glob(f"*12-06*{ext}"))
            im_files.extend(Path(im_dir).glob(f"**/*{ext}"))
        #im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
        include_pdf = False
        if include_pdf:
            pdfs = Path(im_dir).glob(f"**/*.pdf")
            for pdf in tqdm(pdfs):
                outdir = pdf.parent / "pdf2jpg"
                outdir.mkdir(exist_ok=True)
                globbed = list(outdir.glob(f'*{pdf.stem}*.jpg'))
                if len(globbed) == 0:
                    print(f'Converting to jpg: {pdf}')
                    imgsfrompdf = convert_from_path(pdf)
                    to_crop = []
                    for i, img in enumerate(imgsfrompdf):
                        outimg = outdir / f"{pdf.stem}_{i}.jpg"
                        if not outimg.is_file():
                            img.save(outimg, format="jpeg")
                            im_files.append(outimg)
                else:
                    print(f'Already converted to jpg: {pdf}')

        for im in tqdm(sorted(im_files)):
            org_image = cv2.imread(str(im))
            print(org_image.shape)
            # if image isn't in scanned folder and if image hasn't already been scanned, then scan it
            out_img = im.parent / 'scanned' / f"{im.stem}_out.png" if not 'scanned' == str(im.parent.stem) else None
            if not 'scanned' == str(im.parent.stem) and not out_img.is_file():
                    continue
                    #scanned = scanner.scan(im, debug=debug)
            elif 'scanned' == str(im.parent.stem):
                scanned = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
            else:
                print(f'Ignoring {im}')
                continue
            ratio = 1024 / scanned.shape[0]
            org_image = cv2.resize(org_image, (round(org_image.shape[1] * ratio), 1024), interpolation=cv2.INTER_LINEAR)
            scanned = cv2.resize(scanned, (round(scanned.shape[1]*ratio), 1024), interpolation=cv2.INTER_NEAREST)
            # Output numbers of checked boxes
            blur = cv2.GaussianBlur(scanned, (5, 5), 0)
            if debug:
                cv2.imshow("Blur", blur)
                cv2.waitKey()

            thresh2 = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
            if debug:
                cv2.imshow("thresh2", thresh2)
                cv2.waitKey()

            grid_contours, areas, peris = scanner.get_grid_contour(thresh2)

            # cv2.drawContours(org_image, grid_contours, -1, (0, 0, 255), 1)
            # cv2.imshow("Contours", org_image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            #CREATING AOI FROM GRID CONTOURS
            try:
                concat = np.concatenate(grid_contours)
                hull = cv2.convexHull(concat)
                area_hull = cv2.contourArea(hull)
                peri = cv2.arcLength(hull, True)

                x, y, w, h = cv2.boundingRect(concat)
                bbox = np.array([y, x+w, y+h, x])  # top, right, bottom, left
                area_bb = ((x+w)-x)*((y+h)-y)

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
                outdir_trn = Path(f'/home/remi/Documents/sherbrooke_citoyen/training_img_7nov/{im.parent.parent.parent.name}')
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

            except ValueError as e:
                print(e)
                continue
