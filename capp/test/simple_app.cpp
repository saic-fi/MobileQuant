#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>

#include "libllmod.h"

#define ERR(header) if (status) { \
    std::cerr << header << libllmod_get_error_description(status) << "; " << libllmod_get_last_error_extra_info(status, ctx) << std::endl; \
    if (ctx) \
        libllmod_release(ctx); \
    return 1; }

// TODO
#define DEF_MODELS_PATH "/data/local/tmp/llm/"

#ifdef __ANDROID__ 
 #ifdef LIBLLMOD_DEBUG
  #define DEF_LOG_LEVEL LIBLLMOD_LOG_DEBUG
 #else
  #define DEF_LOG_LEVEL LIBLLMOD_LOG_INFO
 #endif
#else
 #define DEF_LOG_LEVEL LIBLLMOD_LOG_DEBUG
#endif



int main(int argc, char *argv[]) {
    void* ctx = nullptr;
    unsigned int log_level = 1; //DEF_LOG_LEVEL;

    std::string models_dir;
    std::string device_type = "8gen3";

    if (argc == 3) {
        device_type = argv[2];
    } 
    
    if (argc >= 2) {
        models_dir = argv[1];
    } else {
        models_dir = DEF_MODELS_PATH;
    }
    
    std::string model_type = "llama";
    if (models_dir.find("gemma") != std::string::npos) {
        model_type = "gemma";
    }
    int use_htp = 1;
    float temperature = 0.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    unsigned long long rng_seed = 1234; // seed rng with time by default
    int max_sequence_length = 1024;
    
    // load the model
    int status = libllmod_setup(
        &ctx, 
        models_dir.c_str(), 
        log_level, 
        device_type.c_str(), 
        use_htp,
        model_type.c_str(), 
        temperature, topp, rng_seed, max_sequence_length
    );
    int steps = 2048;            // number of steps to run for
    int last_token_position = -1;
    char* text = nullptr;

    std::string chat_prefix = "";
    std::string chat_suffix = "";
    std::string welcome = "Hello, how can I help you today?\n>>> ";
    if (model_type == "gemma") {
        welcome = "Please enter the prompt\n>>> ";
    }
    std::string cmd;


    while (true) {
        std::cout << welcome;
        std::getline(std::cin, cmd);
        if (cmd.empty())
            continue;

        if (cmd[0] == '!') { // settings command
            std::istringstream iss(cmd);
            iss >> cmd;
            if (cmd == "!log") {
                unsigned int arg = 0;
                iss >> arg;
                if (ctx != nullptr) {
                    status = libllmod_set_log_level(ctx, arg);
                    ERR("Error setting log level: ")
                }
                log_level = arg;
            } else if (cmd == "!reload") {
                if (ctx != nullptr) {
                    status = libllmod_release(ctx);
                    ERR("Could not release the context: ")
                    ctx = nullptr;
                    std::cout << "Old context released" << std::endl;
                }
                if (iss)
                    iss >> models_dir;

                std::cout << "Loading models from " << models_dir << std::endl;
                std::cout << "Options are:" << std::endl;
                std::cout << "  temperature = " << temperature << std::endl;
                std::cout << "  top-p = " << topp << std::endl;
                std::cout << "  random seed = " << rng_seed << std::endl;
                std::cout << "  number of steps = " << steps << std::endl;
                std::cout << "  system prompt = " << chat_prefix << std::endl; 
                std::cout << "  model type = " << model_type << std::endl;
                std::cout << "  log level (info:2, debug:3) = " << log_level << std::endl;

                status = libllmod_setup(&ctx, models_dir.c_str(), log_level, device_type.c_str(), use_htp,
                                        model_type.c_str(), temperature, topp, rng_seed, max_sequence_length);
                ERR("Initialization errror: ")
                
                std::cout << "Models loaded and ready" << std::endl;
            } else if (cmd == "!exit") {
                break;
            } else {
                std::cerr << "Unknown command: " << cmd << std::endl;
            }
        } 
        else if (cmd[0] == '-') { // args command
            std::istringstream iss(cmd);
            std::vector<std::string> argv;
            std::string arg;

            // -t 1.0 -p 0.9 -s 1234 -n 256 -y Hello, I am a chatbot
            while (std::getline(iss, arg, '-')) {
                if (!arg.empty()) {
                    argv.push_back("-" + arg);
                }
            }

            for (size_t i = 0; i < argv.size(); i+=2) {
                if (argv[i][1] == 't') { temperature = std::stof(argv[i + 1]); }
                else if (argv[i][1] == 'p') { topp = std::stof(argv[i + 1]); }
                else if (argv[i][1] == 's') { rng_seed = std::stoi(argv[i + 1]); }
                else if (argv[i][1] == 'n') { steps = std::stoi(argv[i + 1]); }
                else if (argv[i][1] == 'y') { chat_prefix = argv[i + 1]; }
                else if (argv[i][1] == 'm') { model_type = argv[i + 1]; }
                else { 
                    std::cerr << "Unknown arg, options are:" << std::endl;
                    std::cerr << "  -t <float>  temperature in [0,inf], default 1.0" << std::endl;
                    std::cerr << "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9" << std::endl;
                    std::cerr << "  -s <int>    random seed, default time(NULL)" << std::endl;
                    std::cerr << "  -n <int>    number of steps to run for, default 256" << std::endl;
                    std::cerr << "  -y <string> system prompt in chat mode" << std::endl; 
                    std::cerr << "  -m <string>  model type, default llama" << std::endl;
                }
            }
        } 
        else { // chat or generate
            if (ctx == nullptr) {
                std::cerr << "Please load the models first with '!reload [<models_dir:str>]'" << std::endl;
                continue;
            }

            // gemma template
            // <bos><start_of_turn>user
            // Write a hello world program<end_of_turn>
            // <start_of_turn>model
            // 2 106 user
            // Write a hello world program 107
            // 106 model

            // llama template
            // <|system|>
            // You are a friendly chatbot who always responds in the style of a pirate.</s>
            // <|user|>
            // How many helicopters can a human eat in one sitting?</s>
            // <|assistant|>

            // if (model_type == "llama") {
            //     chat_prefix = "<|user|>\n";  
            //     chat_suffix ="</s>\n<|assistant|>\n";
            // } else {
            //     chat_prefix = "<bos><start_of_turn>user\n";  
            //     chat_suffix ="<end_of_turn>\n<start_of_turn>model\n";
            // }
                        
            // cmd = chat_prefix + cmd + chat_suffix;
            // std::cout << "Prompt: " << cmd << std::endl;

            // status = libllmod_run(ctx, cmd.c_str(), &text, steps, last_token_position);
            status = libllmod_run(ctx, cmd.c_str(), &text, steps, last_token_position);
            ERR("Text generation error:")

            // std::cout << "Finished!" << std::endl;
        } // end of if
        welcome = ">>> ";
    } // end of while

    libllmod_release(ctx);
    delete[] text;
    return 0;
}
