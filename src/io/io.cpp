#include "io.h"

void showImage(const cv::Mat &img, const std::string &title) {
    if (img.empty()) {
        std::cerr << "ERROR: Can't show empty image" << std::endl;
    }

    cv::imshow(title, img);
    cv::waitKey(0);
}

cv::Mat loadImage(const std::string &imagePath) {
    itk::simple::Image vol = itk::simple::ReadImage(imagePath);

    // 1. Extract the middle slice (e.g., slice 75 of the Z-axis)
    std::vector<unsigned int> size = vol.GetSize();
    std::vector<int> start = {0, 0, (int)size[2] / 2};
    std::vector<unsigned int> sliceSize = {size[0], size[1],
                                           0}; // 0 in Z extracts a 2D slice
    itk::simple::Image slice = itk::simple::Extract(vol, sliceSize, start);

    // 2. Rescale intensity to 0-255 and cast to 8-bit for OpenCV
    slice = itk::simple::RescaleIntensity(slice, 0, 255);
    slice = itk::simple::Cast(slice, itk::simple::sitkUInt8);

    // 3. Convert to OpenCV Mat
    // SimpleITK image buffer is contiguous.
    cv::Mat cvImage(cv::Size(slice.GetWidth(), slice.GetHeight()), CV_8UC1,
                    slice.GetBufferAsUInt8());

    // 4. IMPORTANT: Clone the mat if you plan to use it after 'slice' goes out
    // of scope
    cv::Mat result = cvImage.clone();
    return result;
}
