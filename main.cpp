#include <arrayfire.h>

#include <algorithm>
#include <filesystem>
#include <flashlight/fl/flashlight.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <tuple>
#include <vector>

constexpr int PIXEL_MAX = 255;
constexpr int IMG_DIM = 30;
constexpr int INPUT_IDX = 0;
constexpr int TARGET_IDX = 1;
constexpr int TOTAL_CLASSES = 43; // loads 0..num-1 categories
static int TRAIN_SIZE = 0;

// returns imagedim*imagedim vector for a single channel
std::vector<int> mat_to_vec(const cv::Mat& image) {
    std::vector<int> vecdata;
    uint8_t* myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int _stride = image.step; // in case cols != strides
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint8_t val = myData[i * _stride + j];
            vecdata.push_back(static_cast<int>(val));
        }
    }
    return vecdata;
}

std::vector<int> concat_vectors(std::vector<int> a, std::vector<int> b, std::vector<int> c) {
    a.insert(std::end(a), std::begin(b), std::end(b));
    a.insert(std::end(a), std::begin(c), std::end(c));
    return a;
}

std::vector<int> expand_label_vec(std::vector<int>& img_count, int start) {
    std::vector<int> labels;
    for (int i = 0; i < img_count.size(); i++)
        std::fill_n(std::back_inserter(labels), img_count[i], start++);
    return labels;
}

// takes a range for reading images.
// for reading the first two directories 0 and 1, pass 0,2
// for reading from 3,4,5 directories pass 3 and 6
std::pair<af::array, af::array> load_train_images(const std::size_t start, const std::size_t last) {
    namespace fs = std::filesystem;
    const std::string prefix = "../dataset/Train/";
    std::vector<int> images_data;
    std::vector<int> img_count;

    for (int i = start; i < last; i++) {
        int count = 0;
        for (const auto& p : fs::directory_iterator(prefix + std::to_string(i))) {
            cv::Mat3b img = imread(p.path().string(), cv::IMREAD_UNCHANGED); // reading the image
                                                                             // unchanged
            cv::resize(img, img, {IMG_DIM, IMG_DIM}, 0, 0,
                       cv::INTER_NEAREST); // resizing to IMG_DIMxIMG_DIMx3

            if (!img.empty()) {
                cv::Mat1b b, g, r;
                cv::extractChannel(img, b, 0);
                cv::extractChannel(img, g, 1);
                cv::extractChannel(img, r, 2);

                auto image_vec = concat_vectors(mat_to_vec(b.rowRange(0, IMG_DIM)),
                                                mat_to_vec(g.rowRange(0, IMG_DIM)),
                                                mat_to_vec(r.rowRange(0, IMG_DIM)));

                images_data.insert(std::end(images_data), std::begin(image_vec),
                                   std::end(image_vec));
                TRAIN_SIZE += 1;
            }
            ++count;
        }

        img_count.push_back(count);
    }

    std::vector<float> f_images(std::begin(images_data), std::end(images_data));
    images_data.clear();
    std::vector<long long int> dims = {TRAIN_SIZE, IMG_DIM, IMG_DIM, 3};
    std::vector<long long int> rdims(dims.rbegin(), dims.rend());
    af::dim4 af_dims(rdims.size(), rdims.data());

    auto labels = expand_label_vec(img_count, start);
    std::vector<long long int> ldims{TRAIN_SIZE};
    af::dim4 lab_dims(ldims.size(), ldims.data());
    return std::make_pair(af::array(af_dims, f_images.data()), af::array(lab_dims, labels.data()));
}

// loads data of 0,1,... num-1 images and labels
std::pair<af::array, af::array> load_dataset(const int start, const int num) {
    auto [ims, labels] = load_train_images(start, num);
    ims = moddims(ims, {IMG_DIM, IMG_DIM, 3, TRAIN_SIZE});
    ims = ims.T();
    ims = (ims - PIXEL_MAX / 2) / PIXEL_MAX; // Rescale to [-0.5,  0.5]
    return {ims, labels};
}

// data contain name of png along with the relative paths e.g dataset/.../some_image.png
std::pair<af::array, af::array> load_validation_images(const std::vector<std::string>& data) {
    std::vector<int> images_data;
    for (const auto& file : data) {
        cv::Mat3b img = imread(file, cv::IMREAD_UNCHANGED); // reading the image unchanged
        cv::resize(img, img, {IMG_DIM, IMG_DIM}, 0, 0,
                   cv::INTER_NEAREST); // resizing to IMG_DIMxIMG_DIMx3
        if (!img.empty()) {
            cv::Mat1b b, g, r;
            cv::extractChannel(img, b, 0);
            cv::extractChannel(img, g, 1);
            cv::extractChannel(img, r, 2);

            auto image_vec = concat_vectors(mat_to_vec(b.rowRange(0, IMG_DIM)),
                                            mat_to_vec(g.rowRange(0, IMG_DIM)),
                                            mat_to_vec(r.rowRange(0, IMG_DIM)));

            images_data.insert(std::end(images_data), std::begin(image_vec), std::end(image_vec));
        }
    }
    std::vector<float> f_images(std::begin(images_data), std::end(images_data));
    images_data.clear();
    std::vector<long long int> dims = {static_cast<long long int>(data.size()), IMG_DIM, IMG_DIM,
                                       3};
    std::vector<long long int> rdims(dims.rbegin(), dims.rend());
    af::dim4 af_dims(rdims.size(), rdims.data());
    auto ims = af::array(af_dims, f_images.data());

    ims = moddims(ims, {IMG_DIM, IMG_DIM, 3, static_cast<int>(data.size())});
    ims = ims.T();
    ims = (ims - PIXEL_MAX / 2) / PIXEL_MAX; // Rescaling to [-0.5,  0.5]

    // creating fake labels
    std::vector<int> labels(data.size());
    std::iota(std::begin(labels), std::end(labels), 0);

    std::vector<long long int> ldims{static_cast<long long int>(data.size())};
    af::dim4 lab_dims(ldims.size(), ldims.data());

    return std::make_pair(ims, af::array(lab_dims, labels.data()));
}

std::vector<std::string> read_file_names(const std::string& file) {
    std::vector<std::string> files;
    std::ifstream infile(file);
    if (infile) {
        std::string line;
        while (infile) {
            std::getline(infile, line);
            files.push_back(line);
        }
    } else
        throw std::runtime_error("Failed to Read File Names");
    return files;
}

void perform_inference(fl::Sequential& model, const std::string& file) {
    std::cout << "[Performing Inferences]" << std::endl;
    const auto data = read_file_names(file);
    const int batch_size = 1;
    auto [val_x, val_y] = load_validation_images(data);

    std::cout << "Val Dims: " << val_x.dims() << std::endl;

    val_x = val_x(af::span, af::span, 0, af::seq(0, data.size() - 1));
    val_y = val_y(af::seq(0, data.size() - 1));

    fl::BatchDataset valset(
        std::make_shared<fl::TensorDataset>(std::vector<af::array>{val_x, val_y}), batch_size);

    model.eval();

    auto max_element_index = [](af::array vec) {
        std::vector<int> all_data;

        float max = std::numeric_limits<float>::min();
        for (int i = 0; i < TOTAL_CLASSES; i++) {
            all_data.push_back(static_cast<int>(vec(i).scalar<float>()));
        }
        return std::distance(std::begin(all_data),
                             std::max_element(std::begin(all_data), std::end(all_data)));
    };

    std::ofstream out("output.txt");
    std::size_t i = 0;

    for (auto& example : valset) {
        auto inputs = fl::noGrad(example[INPUT_IDX]);
        auto output = model(inputs);

        const int cat = max_element_index(output.array());

        printf("[%s]: %d \n", data[i].c_str(), cat);
        out << "[" << data[i] << "]: " << cat << std::endl;
        i++;
    }
    std::cout << "[Output has been written to out.txt]" << std::endl;
    model.train();
}

std::pair<double, double> eval_loop(fl::Sequential& model, fl::BatchDataset& dataset) {
    fl::AverageValueMeter loss_meter;
    fl::FrameErrorMeter error_meter;

    // Place the model in eval mode.
    model.eval();
    for (auto& example : dataset) {
        auto inputs = fl::noGrad(example[INPUT_IDX]);
        auto output = model(inputs);

        // Get the predictions in max_ids
        af::array max_vals, max_ids;
        af::max(max_vals, max_ids, output.array(), 0);

        auto target = fl::noGrad(example[TARGET_IDX]);

        // Compute and record the prediction error.
        error_meter.add(reorder(max_ids, 1, 0), target.array());

        // Compute and record the loss.
        auto loss = fl::categoricalCrossEntropy(output, target);
        loss_meter.add(loss.array().scalar<float>());
    }

    // Place the model back into train mode.
    model.train();
    double error = error_meter.value();
    double loss = loss_meter.value()[0];
    return std::make_pair(loss, error);
}

int main() {
    fl::init();
    af::setSeed(495);

    constexpr float learning_rate = 1e-3;
    constexpr int epochs = 2;
    constexpr int batch_size = 2048;

    constexpr int CLASSES_RANGE_START = 20;

    constexpr int CLASSES_RANGE_END = 43;

    auto [train_x, train_y] = load_dataset(CLASSES_RANGE_START, CLASSES_RANGE_END);

    const int VAL_SIZE = static_cast<int>(TRAIN_SIZE * 0);

    std::cout << "Total Training Images: " << TRAIN_SIZE << std::endl;
    std::cout << "Train Dims: " << train_x.dims() << std::endl;
    std::cout << "Labels Dims: " << train_y.dims() << std::endl;

    train_x = train_x(af::span, af::span, 0, af::seq(VAL_SIZE, TRAIN_SIZE - 1));
    train_y = train_y(af::seq(VAL_SIZE, TRAIN_SIZE - 1));

    auto val_x = train_x;
    auto val_y = train_y;

    // Make the training batch dataset
    fl::BatchDataset trainset(
        std::make_shared<fl::TensorDataset>(std::vector<af::array>{train_x, train_y}), batch_size);

    // Make the validation batch dataset
    fl::BatchDataset valset(
        std::make_shared<fl::TensorDataset>(std::vector<af::array>{val_x, val_y}), batch_size);

    fl::Sequential model;
    auto pad = fl::PaddingMode::SAME;

    // adding our model
    model.add(fl::View(af::dim4(IMG_DIM, IMG_DIM, 1, -1)));
    model.add(fl::Conv2D(1 /* input channels */, 32 /* output channels */, 5 /* kernel width */,
                         5 /* kernel height */, 1 /* stride x */, 1 /* stride y */,
                         pad /* padding mode */, pad /* padding mode */));
    model.add(fl::ReLU());
    model.add(fl::Pool2D(2 /* kernel width */, 2 /* kernel height */, 2 /* stride x */,
                         2 /* stride y */));
    model.add(fl::Conv2D(32, 64, 5, 5, 1, 1, pad, pad));
    model.add(fl::ReLU());
    model.add(fl::Pool2D(2, 2, 2, 2));

    model.add(fl::View(af::dim4(7 * 7 * 64, -1)));
    model.add(fl::Linear(7 * 7 * 64, 1024));
    model.add(fl::ReLU());
    model.add(fl::Dropout(0.5));

    /////
    model.add(fl::Linear(1024, 2048));
    model.add(fl::ReLU());
    model.add(fl::Dropout(0.25));

    model.add(fl::Linear(2048, 1024));
    model.add(fl::ReLU());
    model.add(fl::Dropout(0.5));

    model.add(fl::Linear(1024, 512));
    model.add(fl::ReLU());
    model.add(fl::Dropout(0.25));

    model.add(fl::Linear(512, 256));
    model.add(fl::ReLU());
    model.add(fl::Dropout(0.5));

    model.add(fl::Linear(256, 1024));
    model.add(fl::ReLU());
    model.add(fl::Dropout(0.25));
    /////

    model.add(fl::Linear(1024, TOTAL_CLASSES));
    model.add(fl::LogSoftmax());

    // Make the optimizer
    fl::AdamOptimizer opt(model.params(), learning_rate);
    // The main training loop
    for (int e = 0; e < epochs; e++) {
        fl::AverageValueMeter train_loss_meter;

        // Get an iterator over the data
        for (auto& example : trainset) {
            // Make a Variable from the input array.
            auto inputs = fl::noGrad(example[INPUT_IDX]);

            // Get the activations from the model.
            auto output = model(inputs);

            // Make a Variable from the target array.
            auto target = fl::noGrad(example[TARGET_IDX]);

            // Compute and record the loss.
            auto loss = fl::categoricalCrossEntropy(output, target);
            train_loss_meter.add(loss.array().scalar<float>());

            // Backprop, update the weights and then zero the gradients.
            loss.backward();
            opt.step();
            opt.zeroGrad();
        }

        double train_loss = train_loss_meter.value()[0];

        // Evaluate on the dev set.
        double val_loss, val_error;
        std::tie(val_loss, val_error) = eval_loop(model, valset);

        std::cout << "Epoch " << e << std::setprecision(3) << ": Avg Train Loss: " << train_loss
                  << " Validation Loss: " << val_loss << " Validation Error (%): " << val_error
                  << std::endl;
    }
    perform_inference(model, "files.txt");
    return 0;
}