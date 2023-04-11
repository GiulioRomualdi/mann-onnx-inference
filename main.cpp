
#include <algorithm>
#include <chrono>
#include <iostream>
#include <onnxruntime_cxx_api.h>

// This is the structure to interface with the MANN model
// After instantiation, set the input_data_ to be the 137 float
// Then call run() to fill in the results_ data
struct MANN
{
    MANN(const std::string& model_path = "mann.onnx")
        : session_(env, model_path.c_str(), Ort::SessionOptions{nullptr})
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
                                                        input_data_.data(),
                                                        input_data_.size(),
                                                        input_shape_.data(),
                                                        input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
                                                         results_.data(),
                                                         results_.size(),
                                                         output_shape_.data(),
                                                         output_shape_.size());
    }

    void run()
    {
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        Ort::RunOptions run_options;
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    }

    // these sizes come from MANN
    static constexpr const int input_size_ = 137;
    static constexpr const int output_size_ = 103;

    std::array<float, input_size_> input_data_{};
    std::array<float, output_size_> results_{};

private:
    Ort::Env env;
    Ort::Session session_;

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 2> input_shape_{1, input_size_};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, output_size_};
};

int main()
{
    std::unique_ptr<MANN> net = std::make_unique<MANN>();

    std::srand(42);
    std::generate(net->input_data_.begin(), net->input_data_.end(), std::rand);

    auto t1 = std::chrono::high_resolution_clock::now();
    net->run();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration<double>(t2 - t1);

    std::cout << "Elapsed time: " << ms_int.count() << "s" << std::endl;

    // std::cerr << "Net input" << std::endl;
    // for (const auto& r : net->input_data_)
    // {
    //     std::cerr << r << ", " ;
    // }
    // std::cerr << std::endl;

    // std::cerr << "Net output" << std::endl;
    // for (const auto& r : net->results_)
    // {
    //     std::cerr << r << ", " ;
    // }
    // std::cerr << std::endl;
}
