from utils import *


if __name__ == '__main__':
    dataset_path = os.path.join('..', 'Images')
    no_classes = 4
    seed = 42

    ### Get every image path in a list
    file_list = get_image_paths(dataset_path, no_classes)

    ### Split the data into train and test
    train_files, test_files = train_test_split(file_list, seed)

    ### Get train and test images and labels
    train_img = [read_image_cv2(file) for folder in train_files for file in folder]
    test_img = [read_image_cv2(file) for folder in test_files for file in folder]

    train_labels = [i for i in range(no_classes) for j in range(len(train_files[i]))]
    test_labels = [i for i in range(no_classes) for j in range(len(test_files[i]))]
    
    ### Resize images
    train_img_resized = list(map(resize_factor_cv2, train_img))
    test_img_resized = list(map(resize_factor_cv2, test_img))
    
    ### Segment images
    train_img_seg = list(map(manual_segmentation, train_img_resized))
    test_img_seg = list(map(manual_segmentation, test_img_resized))
    
    ### Get contours
    train_packed = list(map(get_contour_cv2, train_img_seg, train_img_resized))
    test_packed = list(map(get_contour_cv2, test_img_seg, test_img_resized))
    
    train_contours = [i[0] for i in train_packed]
    train_orig_crop = [i[1] for i in train_packed]
    test_contours = [i[0] for i in test_packed]
    test_orig_crop = [i[1] for i in test_packed]
    
    ### Resize contours
    train_contours_resized = list(map(lambda img: resize_cv2(img, (256, 256)), train_contours))
    train_orig_crop_resized = list(map(lambda img: resize_cv2(img, (256, 256)), train_orig_crop))
    test_contours_resized = list(map(lambda img: resize_cv2(img, (256, 256)), test_contours))
    test_orig_crop_resized = list(map(lambda img: resize_cv2(img, (256, 256)), test_orig_crop))
    
    ### Compute HOG   
    train_hog = list(map(lambda img: compute_hog_scikit(img, nbins=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), axis=-1), train_orig_crop_resized))
    test_hog = list(map(lambda img: compute_hog_scikit(img, nbins=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), axis=-1), test_orig_crop_resized))
    
    train_hog = np.asanyarray(train_hog)
    test_hog = np.asanyarray(test_hog)
    
    ### Train and test classifiers
    print("SVM")
    svm_classifier(train_hog, train_labels, test_hog, test_labels)

    print("KNN 2 neighbors")
    knn_classifier(train_hog, train_labels, test_hog, test_labels)

    


    
    
    
    
    
    






    