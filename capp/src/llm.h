#ifndef LIBLLMOD_LLM_H
#define LIBLLMOD_LLM_H

#include <string>
#include <memory>
#include "context.h"

namespace libllmod {

#define mask_dtype  uint16_t
#define kv_dtype    uint8_t

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling


typedef struct {
    int vocab_size;
    std::vector<ProbIndex> probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// ----------------------------------------------------------------------------
// from https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L19
typedef struct {
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
    int rope_base;
} Config;

// the embedding layer is not compiled in the model
typedef struct {
    std::span<float> token_embedding_table; // (vocab_size, dim)
} TransformerWeights;


// ----------------------------------------------------------------------------
// llama2.c tokenizer
// from https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L365C1-L365C77

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;


// ----------------------------------------------------------------------------
class LLM: public Context {
public:
    LLM(
        std::string const& models_dir, 
        std::string const& model_type, 
        LogLevel log_level, std::string const& device_type, 
        bool use_htp=true,
        float temperature=1.0f, float topp=0.9f, 
        unsigned long long rng_seed=0, 
        int max_sequence_length=1024
    );
    virtual ~LLM();

    void forward(int token, int pos);
    int sample();

    void prepare_buffers() override;
    void generate(const char* prompt, int steps, char** text_out, int& last_token_position) override;
    virtual void run() override;

private:
    Tokenizer tokenizer;
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    Sampler sampler;
    std::string _model_type;
    std::vector<int> _chat_prefix;
    std::vector<int> _chat_suffix;

    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes

    // rope cache
    std::vector<float> cos_cache; // (max_sequence_length, n_elements)
    std::vector<float> sin_cache; // (max_sequence_length, n_elements)
    void build_rope_cache(int max_sequence_length, int head_dim, int base=10000);

    // input and output buffers in float
    std::vector<float> input_feat;
    std::vector<float> input_cos;
    std::vector<float> input_sin;
    std::vector<float> logits; // (1 x 32000)

    // // stop tokens
    // int* stop_tokens = (int*)malloc(3 * sizeof(int));
    // int n_stop_tokens;
};


} // end of namespace libllmod

#endif // LIBLLMOD_LLM_H
