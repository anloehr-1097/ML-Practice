#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/dist.h>
#include <ATen/ops/rand.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>
#include <iostream>
#include <tuple>


class Encoder: torch::nn::Module {
public:
    int hidden_dim {0};
    std::tuple<int, int> inp_dim {};
    //= std::make_tuple(28, 28);


    Encoder (const int dim_h,
             const std::tuple<int, int> &size_input_image){
        hidden_dim = dim_h;
        inp_dim = size_input_image;
    };

    torch::Tensor forward(torch::Tensor x){
        return 2 * x;
    };
};


class Decoder: torch::nn::Module {
public:
    int final_latent_size {0};

    Decoder (const int fin_dim_latent){
        final_latent_size = fin_dim_latent;
        auto l_flatten = register_module("l_flatten", torch::nn::Flatten());
    }
             
    torch::Tensor forward(torch::Tensor x){
        return x;
    };
};

class VAE: torch::nn::Module {
public:
    Encoder *enc;
    Decoder *dec;

    VAE(Encoder *encoder, Decoder *decoder) {
        enc = encoder;
        dec = decoder;

    };
    VAE(int n_hidden, std::tuple<int, int> inp_size);

    torch::Tensor forward(torch::Tensor x) {
        return x;
    };

    torch::Tensor encode(torch::Tensor x) {
        ;
        return x;
    };

    torch::Tensor decode(torch::Tensor x) {
        ;
        return x;
    };

};


int main() {
    Encoder enc(100, std::make_tuple(28, 28));
    Decoder dec (10);
    VAE vae(&enc, &dec);

    torch::Tensor tensor_a = torch::rand({2, 3});
    std::cout << tensor_a << std::endl;
    torch::Tensor out = enc.forward(tensor_a);
    std::cout << out << std::endl;

    torch::Tensor vae_out = vae.forward(tensor_a);
    std::cout << vae_out << std::endl;

    // torch::Tensor tensor_b = torch::rand({3, 3});
    // std::cout << tensor_b << std::endl;
    // torch::Tensor tensor_norm = torch::normal(0.0  , 1.0, 10);
    // std::cout << tensor_norm << std::endl;

}
