#include "llm.h"
#include "error.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <span>
#include <cassert>


using namespace libllmod;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// llama2.c tokenizer 
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}


void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "failed read max token length\n"); 
        exit(EXIT_FAILURE); 
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { 
            fprintf(stderr, "failed read %d vocab score.\n", i); 
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) { 
            fprintf(stderr, "failed read len.\n"); exit(EXIT_FAILURE); 
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { 
            fprintf(stderr, "failed read %d vocab. \n", i); 
            exit(EXIT_FAILURE); 
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}


void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}


char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    // if (prev_token == bos && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}


void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}


int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}


void encode(Tokenizer* t, const char *text, const std::vector<int>& prefix, const std::vector<int>& suffix, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (libllmod::TokenIndex *) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char *) malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    // if (bos != 0) tokens[(*n_tokens)++] = bos;
    for (int i = 0; i < prefix.size(); ++i) {
        tokens[(*n_tokens)++] = prefix[i];
    }

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    
    // // std::string whitespace(" ");
    // if (text[0] != '\0') {
    //     char whitespace[ ] = {" "};
    //     int dummy_prefix = str_lookup(whitespace, t->sorted_vocab, t->vocab_size);
    //     tokens[(*n_tokens)++] = dummy_prefix;
    // }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according to the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    // if (eos != 0) tokens[(*n_tokens)++] = eos;
    for (int i = 0; i < suffix.size(); ++i) {
        tokens[(*n_tokens)++] = suffix[i];
    }

    free(str_buffer);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void softmax(std::vector<float>& x) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < x.size(); i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < x.size(); i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < x.size(); i++) {
        x[i] /= sum;
    }
}

// ----------------------------------------------------------------------------
// sampler functions
int sample_argmax(const std::vector<float>& probabilities) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < probabilities.size(); i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}


int sample_mult(const std::vector<float>& probabilities, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < probabilities.size(); i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return probabilities.size() - 1; // in case of rounding errors
}


int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}


unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}


float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}


// ----------------------------------------------------------------------------
void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier, std::size_t ptr_len) {
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->token_embedding_table = { fptr, static_cast<std::size_t>( p->vocab_size * p->dim ) };
}


void read_meta(const char* checkpoint, Config* config, TransformerWeights* weights, 
        int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }

    // read in the version number (uint32), has to be 1
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 1) { fprintf(stderr, "Bad version %d, need version 1\n", version); exit(EXIT_FAILURE); }

    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }

    int header_size = 256; // the header size for version 2 in bytes

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }

    *data = static_cast<float*>(mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0));
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_classifier, *file_size);
}


// The LLM class that wraps the Tokenizer and Transformer objects
LLM::LLM(
    std::string const& models_dir, 
    std::string const& model_type, 
    LogLevel log_level, 
    std::string const& device_type, 
    bool use_htp,
    float temperature, 
    float topp, 
    unsigned long long rng_seed, 
    int max_sequence_length
) : Context(models_dir, log_level, device_type, use_htp) {

    _model_type = model_type;
    // hardcode the tokenized chat template, ugly but work
    if (_model_type == "llama") {
        int prefix[6] = {29966, 29989, 1792, 29989, 29958, 13};
        int suffix[9] = {2, 13, 29966, 29989, 465, 22137, 29989, 29958, 13};
        std::copy(prefix, prefix+6, std::back_inserter(_chat_prefix)); 
        std::copy(suffix, suffix+9, std::back_inserter(_chat_suffix)); 
    } else {
        int prefix[2] = {2, 106};
        int suffix[2] = {107, 106};
        std::copy(prefix, prefix+2, std::back_inserter(_chat_prefix)); 
        std::copy(suffix, suffix+2, std::back_inserter(_chat_suffix)); 
    }
    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;

    // read in the Config and the Weights from the checkpoint
    std::string checkpoint_path = models_dir + "/meta.bin";
    read_meta(checkpoint_path.c_str(), &config, &weights, &fd, &data, &file_size);

    // override max_sequence_length
    config.seq_len = max_sequence_length;
    config.rope_base = 10000;

    // load the tokenizer
    std::string tokenizer_path = models_dir + "/tokenizer.bin";
    // hack for llama2.c tokenizer
    // build_tokenizer(&tokenizer, tokenizer_path.c_str(), config.vocab_size - config.vocab_size % 10);
    build_tokenizer(&tokenizer, tokenizer_path.c_str(), config.vocab_size);

    // build the sampler
    sampler.vocab_size = config.vocab_size;
    sampler.temperature = temperature;
    sampler.topp = topp;
    sampler.rng_state = rng_seed;
    sampler.probindex.resize(config.vocab_size); 

    // build the rope cache
    build_rope_cache(config.seq_len, config.dim / config.n_heads, config.rope_base);

    // std::string stop_word = "\n";
    // encode(&tokenizer, stop_word.c_str(), 1, 0, stop_tokens, &n_stop_tokens);
    // encode(&tokenizer, stop_word.c_str(), 0, 0, stop_tokens, &n_stop_tokens);

#ifdef LIBLLMOD_DEBUG
    std::cout << "DEBUG: config, weights loaded from " << checkpoint_path << std::endl;
    std::cout << "DEBUG: config:" << std::endl;
    std::cout << "DEBUG:     config.dim = " << config.dim << std::endl;
    std::cout << "DEBUG:     config.hidden_dim = " << config.hidden_dim << std::endl;
    std::cout << "DEBUG:     config.n_layers = " << config.n_layers << std::endl;
    std::cout << "DEBUG:     config.n_heads = " << config.n_heads << std::endl;
    std::cout << "DEBUG:     config.n_kv_heads = " << config.n_kv_heads << std::endl;
    std::cout << "DEBUG:     config.vocab_size = " << config.vocab_size << std::endl;
    std::cout << "DEBUG:     config.seq_len = " << config.seq_len << std::endl;
    std::cout << std::endl;
    std::cout << "DEBUG:     tok_embeddings[:10] = " << config.hidden_dim << std::endl;
    for (int i=0; i < 10; i++) {
        std::cout << weights.token_embedding_table[i] << " ";
    }
    std::cout << std::endl;
#endif
}


LLM::~LLM() {
    sampler.probindex.clear();
    free_tokenizer(&tokenizer);
    // free(stop_tokens);
    // stop_tokens = NULL;

    // weights: close the memory mapping 
    if (data != MAP_FAILED) { munmap(data, file_size); }
    if (fd != -1) { close(fd); }
}


void LLM::prepare_buffers() {
    int head_dim = int(config.dim / config.n_heads);

    // input 0: input_feat
    input_feat.resize(config.dim);

    // input 1: input_mask
    mask_dtype* input_mask = reinterpret_cast<mask_dtype*>(inputs[1].get_data_ptr());
    unsigned int seq_len = inputs[1].get_num_elements(1); // TODO
    std::memset((void*)input_mask, 0, seq_len * sizeof(mask_dtype));
    input_mask[seq_len-1] = std::numeric_limits<mask_dtype>::max(); // EOS token

    // input 2&3: input_cos, input_sin
    input_cos.resize(head_dim);
    input_sin.resize(head_dim);

    // input 4&5: kv caches
    int cache_size = config.n_layers * config.n_kv_heads * (config.seq_len-1) * head_dim;
    std::memset(inputs[4].get_data_ptr(), 0, cache_size * sizeof(kv_dtype));
    std::memset(inputs[5].get_data_ptr(), 0, cache_size * sizeof(kv_dtype));

    // logits buffer 
    logits.resize(config.vocab_size);
    debug("LLM fp buffers: logits.size = {}, input_feat = {}, input_cos = {}", logits.size(), input_feat.size(), input_cos.size());
}


void LLM::forward(int token, int pos) {
    if (pos >= config.seq_len) {
        throw libllmod_exception(ErrorCode::INVALID_ARGUMENT, "too many tokens", "Llama::forward", __FILE__, STR(__LINE__));
    }

    // std::chrono::high_resolution_clock::time_point tick = std::chrono::high_resolution_clock::now();
    
    // embedding
    float* content_row = weights.token_embedding_table.data() + token * config.dim;
    // rope cache
    int head_dim = config.dim / config.n_heads;
    float* cos_row = cos_cache.data() + pos * head_dim;
    float* sin_row = sin_cache.data() + pos * head_dim;

    // update inputs: copy(source.begin, source.end, dest.begin)
    std::copy(content_row, content_row + config.dim, input_feat.begin());
    std::copy(cos_row, cos_row + head_dim, input_cos.begin());
    std::copy(sin_row, sin_row + head_dim, input_sin.begin());

    // auto&& tock = std::chrono::high_resolution_clock::now();
    // auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    // printf("copying inputs takes %f ms. \n", (double)(diff.count()));
    
    run();
}


void LLM::run() {
    if (!_qnn_initialized)
        return;

    auto&& burst_scope_guard = scope_guard([this](){ _qnn->start_burst(); }, [this]() { _qnn->end_burst(); });
    (void)burst_scope_guard;

    auto&& gitr = graphs.begin();

    // std::chrono::high_resolution_clock::time_point tick = std::chrono::high_resolution_clock::now();
    inputs[0].set_data(input_feat);
    inputs[2].set_data(input_cos); 
    inputs[3].set_data(input_sin); 
    // inputs[5].set_data(v_cache);
    // auto&& tock = std::chrono::high_resolution_clock::now();
    // auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    // printf("copying inputs 2 takes %f ms. \n", (double)(diff.count()));

    // tick = std::chrono::high_resolution_clock::now();
    gitr->execute();
    // tock = std::chrono::high_resolution_clock::now();
    // diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    // printf("execute graph takes %f ms. \n", (double)(diff.count()));

    // tick = std::chrono::high_resolution_clock::now();
    outputs[0].get_data(logits);
    // outputs[2].get_data(output_v); 
    // tock = std::chrono::high_resolution_clock::now();
    // diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    // printf("copying outputs 2 takes %f ms. \n", (double)(diff.count()));
}


void LLM::generate(const char* prompt, int steps, char** text_out, int& pos) {
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+20) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    // encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    encode(&tokenizer, prompt, _chat_prefix, _chat_suffix, prompt_tokens, &num_prompt_tokens);
    // std::cout << "prompt: " << std::endl;
    // for (int i=0; i < num_prompt_tokens; ++i) {
    //     std::cout << prompt_tokens[i] << " " << decode(&tokenizer, 0, prompt_tokens[i]) << std::endl;
    // }
    
    int next = -100;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int head_dim = config.dim / config.n_heads;

    // std::vector<char> decoded_text;
    int current_step = 0;
    std::chrono::high_resolution_clock::time_point tick = std::chrono::high_resolution_clock::now();
    
    while (current_step < steps) {
        // forward the transformer to get logits for the next token
        pos = (pos + 1) % (config.seq_len-1);
        forward(token, pos); 

        // tick = std::chrono::high_resolution_clock::now();

        // update the buffers
        reinterpret_cast<mask_dtype*>(inputs[1].get_data_ptr())[pos] = std::numeric_limits<mask_dtype>::max(); // unmask the current position
        
        kv_dtype* k_cache_ptr   = (kv_dtype*) inputs[4].get_data_ptr();
        kv_dtype* output_k_ptr  = (kv_dtype*) outputs[1].get_data_ptr();

        kv_dtype* v_cache_ptr   = (kv_dtype*) inputs[5].get_data_ptr();
        kv_dtype* output_v_ptr  = (kv_dtype*) outputs[2].get_data_ptr();

        for (int i = 0; i < config.n_layers; i++) {
            for (int j = 0; j < config.n_kv_heads; j++) {
                int output_offset = (i * config.n_kv_heads + j) * head_dim;
                int cache_offset = (i * config.n_kv_heads + j) * (config.seq_len-1) * head_dim;
                std::memcpy(
                    (void*)(k_cache_ptr + cache_offset + pos * head_dim), 
                    (void*)(output_k_ptr + output_offset), 
                    head_dim*sizeof(kv_dtype)
                );
                std::memcpy(
                    (void*)(v_cache_ptr + cache_offset + pos * head_dim), 
                    (void*)(output_v_ptr + output_offset), 
                    head_dim*sizeof(kv_dtype)
                );
                
            }
        }
        // auto&& tock = std::chrono::high_resolution_clock::now();
        // auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
        // printf("copying outputs takes %f ms. \n", (double)(diff.count()));

        // advance the state machine
        if (current_step < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[current_step + 1]; // NOTE: we waste #prompts forwards but it's necessary for kv cache
        } else {
            // otherwise sample the next token from the logits
            // tick = std::chrono::high_resolution_clock::now();
            next = sample();
            // tock = std::chrono::high_resolution_clock::now();
            // diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
            // printf("samping takes %f ms. \n", (double)(diff.count()));

            // tick = std::chrono::high_resolution_clock::now();
            char* piece = decode(&tokenizer, token, next);
            // tock = std::chrono::high_resolution_clock::now();
            // diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
            // printf("decoding takes %f ms. \n", (double)(diff.count()));
            
            // tick = std::chrono::high_resolution_clock::now();
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
            // tock = std::chrono::high_resolution_clock::now();
            // diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
            // printf("printing %f ms. \n", (double)(diff.count()));
            // break;

            // append token 
            // decoded_text.push_back(piece[0]);

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            // hard-coded: 13 == "\n"
            // if (next == 1 || next == stop_tokens[0] || (next == 13 && current_step > num_prompt_tokens)) { break; }
            if (next == 1 || next == 2 || next == 0) { break; }
        }

        

        token = next;
        // init the timer here because the first iteration can be slower
        if (current_step == 0) { tick = std::chrono::high_resolution_clock::now(); }
        current_step++;
    }

    if (current_step > 1) {
        auto&& tock = std::chrono::high_resolution_clock::now();
        auto&& diff = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
        info("Achieved tok/s: {}", (current_step-1) / (double)((double)(diff.count())) *1000);
        printf("(%f tok/s)\n", (current_step-1) / (double)((double)(diff.count())) *1000);
    }

    // *text_out = new char[decoded_text.size() + 1]; // dynamically allocate memory at address *text_out
    // std::strcpy(*text_out, &decoded_text[0]);
    free(prompt_tokens);
}


int LLM::sample() {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler.temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits);
    } else {
        // apply the temperature to the logits
        for (int q=0; q < logits.size(); q++) { 
            logits[q] /= sampler.temperature; 
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&(sampler.rng_state));
        // we sample from this distribution to get the next token
        if (sampler.topp <= 0 || sampler.topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            //next = sample_topp(logits, topp, probindex, coin);
            next = sample_argmax(logits);
        }
    }
    return next;
}


// ----------------------------------------------------------------------------
void LLM::build_rope_cache(int max_sequence_length, int head_dim, int base) {
    int half_dim = head_dim / 2;
    cos_cache.resize(max_sequence_length * head_dim);
    sin_cache.resize(max_sequence_length * head_dim);

    for (int i = 0; i < max_sequence_length; i++) {
        for (int j = 0; j < head_dim; j++) {
            int k = j % half_dim;
            float theta = (float) (i) / (float) pow(base, (2 * k) / (float) head_dim);
            cos_cache[i * head_dim + j] = cosf(theta);
            sin_cache[i * head_dim + j] = sinf(theta);
            if (j < half_dim) {
                sin_cache[i * head_dim + j] *= -1;
            }
        }
    }
}