import skimage as ski
from kneed import KneeLocator
from skimage.transform import ProjectiveTransform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from pathlib import Path
from skimage.measure import ransac

def load_image(filepath: str) -> np.ndarray:
    return ski.io.imread(filepath)


def save_image(filepath: str, im: np.ndarray):
    ski.io.imsave(filepath, im)


def histogram_equalization(im: np.ndarray) -> np.ndarray:
    equalized = np.stack(
        [ski.exposure.equalize_hist(im[:, :, c]) for c in range(im.shape[2])], axis=-1
    )
    return (equalized * 255).astype(np.uint8)


def grayscale(im: np.ndarray) -> np.ndarray:
    return (ski.color.rgb2gray(im) * 255).astype(np.uint8)


def bin(im: np.ndarray) -> np.ndarray:
    thresh = ski.filters.threshold_otsu(im)
    return ((im > thresh) * 255).astype(np.uint8)


def preprocess(im: np.ndarray,val=False) -> np.ndarray:
    if im.shape[2] == 4:
        im = im[:, :, :3]

    eqalized = histogram_equalization(im)

    gray = grayscale(eqalized)
    if val :
        return bin(gray)
    return gray


def sobel(im: np.ndarray) -> np.ndarray:
    im = ski.filters.sobel(im)
    return (im * 255).astype(np.uint8)


def denoising(im: np.ndarray) -> np.ndarray:
    im = ski.morphology.closing(im)
    im = ski.morphology.opening(im)
    # im = ski.morphology.erosion(im)
    # im = ski.morphology.dilation(im)
    return (im * 255).astype(np.uint8)


def labels(im: np.ndarray) -> np.ndarray:
    return ski.measure.label(im, connectivity=1)


def draw_regions(im_original: np.ndarray, labels: np.ndarray, regions: np.ndarray) -> np.ndarray:
    overlay = ski.color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    overlay = (overlay * 255).astype(np.uint8)
    for region in regions:
        if region.area < 500:
            continue
        minr, minc, maxr, maxc = region.bbox
        rr, cc = ski.draw.rectangle_perimeter(
            (minr, minc), (maxr, maxc), shape=overlay.shape
        )
        overlay[rr, cc] = [255, 0, 0]
        im_original[rr, cc] = [255, 0, 0]
    return overlay, im_original



def draw_squares(im_original: np.ndarray, squares):
    img = im_original.copy()

    for minr, minc, maxr, maxc in squares:
        rr, cc = ski.draw.rectangle_perimeter(
            start=(minr, minc),
            end=(maxr, maxc),
            shape=img.shape
        )
        img[rr, cc] = [0, 255, 0]

    return img

def draw_form(im_original: np.ndarray,coords):
    img = im_original.copy()

    for leftL,leftH,rightL,rightH in coords:
        rr, cc = ski.draw.polygon_perimeter(
            r=np.array([leftL[0],leftH[0],rightL[0],rightH[0]]),
            c=np.array([leftL[1],leftH[1],rightL[1],rightH[1]]),shape=img.shape)
        img[rr, cc] = [0, 255, 0]

    return img

# def rotate(theta:float, )

def get_square_coordinate(image_original:np.ndarray,region):
    # je récupère les coordonnée de la forme
    # je récupère ses points de cornet en utilisant harris
    image_shape = image_original.shape
    mask = np.zeros(shape=(image_shape[0],image_shape[1]))
    mask[region.coords] = 1
    return ski.feature.corner_peaks(ski.feature.corner_harris(mask),min_distance=5)

def find_squares(im_original: np.ndarray, regions: np.ndarray) -> np.ndarray:
    squares = []
    for region in regions:
        if region.area < 500:
            continue
        if region.extent < 0.5:
            continue

        coords = get_square_coordinate(im_original,region)
        if (len(coords) != 4):
            continue
        left_low = coords[0]
        left_high = coords[2]
        right_low = coords[1]
        right_high = coords[3]
        # if (abs(minc-maxc) - 10) <= abs(minr-maxr) and abs(minr-maxr) <= (abs(minc-maxc) + 10):
        squares.append((left_low,left_high,left_low,right_low,right_high))

    res = draw_squares(im_original, squares)
    # print(len(squares))
    return squares, res

def verif_square(im: np.ndarray, squares: np.ndarray) -> []:
    res = []
    for minr, minc, maxr, maxc in squares:
        square = im[minr:maxr, minc:maxc]
        e4 = ski.measure.euler_number(square, connectivity=1)
        object_nb_4 = ski.measure.label(square, connectivity=1).max()
        holes_nb_4 = object_nb_4 - e4
        # print(f'e4:{e4}, nb obj:{object_nb_4}, holes:{holes_nb_4}')
        if holes_nb_4 == 1 and object_nb_4 == 1:
            res.append((minr, minc, maxr, maxc))
    return res


def get_angle(p1,p2,p3):
    d_p1_p2 = p2 - p1
    d_p1_p3 = p3 - p1
    norm_d1 = np.linalg.norm(d_p1_p2)
    norm_d2 = np.linalg.norm(d_p1_p3)
    if norm_d1 == 0 or norm_d2 == 0:
        return 0
    cos = np.dot(d_p1_p2, d_p1_p3) / (norm_d2 * norm_d1)
    return np.degrees(np.arccos(cos))

def filter_qr_triplets(markers: list, max_distance: float = 500.0,limit_angle:float = 20) -> list:
    if len(markers) < 3:
        return []
    centers = np.array(markers)
    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    triplets = []
    value = set()
    n = len(markers)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p_i = centers[i]
                p_j = centers[j]
                p_k = centers[k]

                d_i_j = np.linalg.norm(p_i - p_j)
                d_i_k = np.linalg.norm(p_i - p_k)
                d_j_k = np.linalg.norm(p_j - p_k)
                if d_i_j > max_distance or d_i_k > max_distance:
                    continue
                angle = get_angle(p_i,p_j,p_k)
                if np.abs(angle - 90) > limit_angle:
                    continue
                val = (i,min(j,k),max(j,k))
                if val not in value:
                    value.add(val)
                    triplets.append(val)
    return triplets

def get_element_coordinate(triplet_coordinate):
    top_left = triplet_coordinate[0]
    top_right = triplet_coordinate[2]
    bottom_left = triplet_coordinate[1]
    if abs(triplet_coordinate[1][1] - triplet_coordinate[0][1])  > abs(triplet_coordinate[2][1] - triplet_coordinate[0][1]):
        top_right, bottom_left  = triplet_coordinate[1], triplet_coordinate[2]
    bottom_right = top_right + bottom_left - top_left
    return np.float32([top_left, top_right, bottom_left,bottom_right])

def get_element_drawned(im:np.ndarray,triplet_coordinate):
    #TODO
    return
"""
A rajouter
    1. Morphology traitement : fait
    2.1 Keypoint and descriptor (ORB): fait
    2.2 Matching  : fait
    3. Clustering des points: (DBSCAN ou Kmeans) : fait
    4. RANSAC : fait
    5. triplet à verifier
    6. 4ième point : fait 
"""


def process_directory(image_path: str,mdl_img, mdl_img_process,mdl_kpts,model_desc,orb):
    im = load_image(image_path)
    res = preprocess(im)
    # res = denoising(res)
    k = 4
    marqueur_detected = []

    h,w =mdl_img.shape[0] ,mdl_img.shape[1]
    model_quad = np.float32([[0, 0],[0,w],[h,w],[h,0]])

    ### keypoint and descriptor of the actuel picture
    orb.detect_and_extract(res)
    frame_kpts , frame_descr = orb.keypoints,orb.descriptors
    print("frame_kpts",len(frame_kpts))
    ###  Matching: Ratio test (Simulation de FLAN avec neighbors et Ratio test)
    matches = NearestNeighbors(n_neighbors=k,metric='hamming')
    matches.fit(model_desc)

    distances,indice = matches.kneighbors(frame_descr)
    print("distance, indice",distances,indice)
    print(distances[:, 0].min(), distances[:, 0].mean(), distances[:, 0].max())
    good_matches = []
    T = 0.75
    for  i in range(len(distances)):
        m,n = distances[i][0],distances[i][1]
        i1 = indice[i][0]
        if  m < n * T:
            good_matches.append((i,i1))
    print("good_matches trouvée",len(good_matches))
    pts_mdl = []
    pts_frame = []
    if len(good_matches) < k:
        return
    for i,j in good_matches:
        pts_frame.append(frame_kpts[i])
        pts_mdl.append(mdl_kpts[j])
    pts_mdl = np.array(pts_mdl)
    pts_frame = np.array(pts_frame)
    # clustering (DBSCAN)
    nearest_neghtboors = NearestNeighbors(n_neighbors=k)
    nearest_neghtboors.fit(pts_frame)
    distance_frame,_  = nearest_neghtboors.kneighbors(pts_frame);
    distances_sorted = np.sort(distance_frame[:,k-1],axis=0)
    x = np.arange(len(distances_sorted))
    kneedle = KneeLocator(x,distances_sorted,curve='convex',direction='increasing')
    eps = 20
    if kneedle.knee is not None:
        candidate = distances_sorted[kneedle.knee]
        if  0 < candidate and candidate < 100:
            eps = candidate
            print("bon candidat d'eps:",candidate)
        else:
            print("mauvais candidat d'eps:",candidate)
    else:
        print("pas de candidat trouvée eps :",eps)
    db =  DBSCAN(eps=eps, min_samples=k-1)
    db.fit(pts_frame)
    db_labels = db.labels_
    for label in set(db_labels):
        if label == -1: # éléments qui n'ont pas de labe (label = -1)
            continue
        condition = db_labels == label
        pts_frame_actual = pts_frame[condition]
        pts_mdl_actual = pts_mdl[condition]
        if len(pts_mdl_actual) <k:
            print("pas assez de point de frame actuel")
            continue

        # RANSAC
        model, inliers = ransac((pts_mdl_actual,pts_frame_actual),ProjectiveTransform,min_samples=k,residual_threshold=5,max_trials=1000)
        if model is None:
            print("model a échouer pour Ransac")
            continue
        matches_ransac_inliers = np.array(good_matches)[condition][inliers]
        print("pts de frame actuel : ",pts_frame_actual)
        print("pts de mdl actuel : ",pts_mdl_actual)
        frame_coord = np.round(model(model_quad)).astype(int)
        marqueur_detected.append(frame_coord)
    print("marqueur détectée: ",marqueur_detected)
    res = draw_form(im,marqueur_detected)
    output_dir = Path("../../documents/results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{Path(image_path).stem}_res.png"
    save_image(str(output_path), res)
    # save_image("res.png", res_draw)

def main():
    data_dir = Path("../../data")

    # model preparation
    mdl_img = load_image("../model/frame.png")
    mdl_img_preprocess = preprocess(mdl_img,True)
    save_image("../model/debug_mdl_preprocess.png", mdl_img_preprocess)

    # keypoint and descriptors
    orb = ski.feature.ORB(n_keypoints=3000,n_scales=8,fast_threshold=0.3)
    orb.detect_and_extract(mdl_img_preprocess)
    model_kpts, model_desc = orb.keypoints,orb.descriptors
    print("model kpts",len(model_kpts))
    for image_file in data_dir.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            process_directory(str(image_file),mdl_img,mdl_img_preprocess,model_kpts,model_desc,orb)
    # process_directory("../data/Pastedimage.JPG")

main()
