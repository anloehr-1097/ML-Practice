#include <ATen/Functions.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/dist.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sqrt.h>
#include <cassert>
#include <cmath>
#include <ostream>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/enum.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
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


class TransformerLayer: torch::nn::Module {
    /* 
     * A single transformer module impl. self attention
     * */
private:
    torch::nn::Linear W_k;
    // torch::Tensor W_k;
    torch::nn::Linear  W_v;
    torch::nn::Linear  W_q;
    torch::nn::Linear  W_o;
    torch::nn::LayerNorm ln_self_att = nullptr;
    torch::nn::LayerNorm ln_ff = nullptr;

public:
    int d_k, d_v, d_out, d_in;
    TransformerLayer(int d_out, int d_inp, int d_k, int d_v)
    : W_k(register_module("W_k", torch::nn::Linear(d_inp, d_k))),
    W_q(register_module("W_q", torch::nn::Linear(d_inp, d_k))),
    W_v(register_module("W_v", torch::nn::Linear(d_inp, d_v))),
    W_o(register_module("W_o", torch::nn::Linear(d_v, d_out))),
    ln_self_att(torch::nn::LayerNormImpl({d_v})),
    ln_ff(torch::nn::LayerNormImpl({d_out}))
    {
        this -> d_k = d_k;
        this -> d_v = d_v;
        this -> d_in = d_inp;
        this -> d_out = d_out;
    };

    void print(){std::cout << W_k << std::endl << W_q << std::endl << W_v << std::endl << W_o << std::endl; };

    torch::Tensor forward(torch::Tensor x){

        torch::Tensor k, q, v, weighting, combi_val;
        k = W_k(x);   // dim: input_length x d_k
        q = W_q(x);  // dim: input_length x d_k
        v = W_v(x);  // size: inp_length x d_v

        // weighting = q * k / sqrt(double (d_k));

        std::cout << q.sizes();
        std::cout << k.t().sizes() << std::endl;
        weighting = torch::matmul(q, k.t());
        weighting /= sqrt((double) (d_k));
        std::cout << "d_k" << d_k << std::endl;
        std::cout << "Weighting: "<< weighting << std::endl;
        std::cout << "Weighting size: "<< weighting.sizes() << std::endl;
        weighting = torch::softmax(weighting, 0);
        std::cout << weighting;
        combi_val = torch::matmul(weighting, v);  // attention output, size inp_length x d_v
        // add & apply layer norm
        combi_val += x;
        combi_val = ln_self_att -> forward(combi_val);

        std::cout << "Combi val:" << combi_val;
        std::cout << combi_val.sizes();
        std::cout << "Combi val after add and layer norm self attention: " << combi_val << std::endl;

        torch::Tensor out = ln_ff -> forward(W_o(combi_val) + combi_val);
        return out;
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
    std::cout << "Hello invoke 01/31/25 \n";

    torch::Tensor x = torch::randn({3, 4});  // inp_length, inp_dim
    // torch::nn::Linear lin1 = torch::nn::Linear(4,10);
    // torch::nn::LayerNormImpl ln1 = torch::nn::LayerNormImpl({10});
    // std::cout << ln1.forward(lin1(x));

    TransformerLayer trans(4, 4, 2, 4);
    torch::Tensor trans_out = trans.forward(x);
    std::cout << "Trans out: ";
    trans_out.print();
    



    // torch::Tensor tensor_b = torch::rand({3, 3});
    // std::cout << tensor_b << std::endl;
    // torch::Tensor tensor_norm = torch::normal(0.0  , 1.0, 10);
    // std::cout << tensor_norm << std::endl;

}
