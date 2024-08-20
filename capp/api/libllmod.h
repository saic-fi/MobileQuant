#ifndef LIBLLMOD_API
#define LIBLLMOD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif


enum libllmod_status_code {
      LIBLLMOD_NO_ERROR,
      LIBLLMOD_INVALID_CONTEXT,
      LIBLLMOD_INVALID_ARGUMENT,
      LIBLLMOD_FAILED_ALLOCATION,
      LIBLLMOD_RUNTIME_ERROR,
      LIBLLMOD_INTERNAL_ERROR,
};


enum libllmod_log_level {
   	LIBLLMOD_LOG_NOTHING,
   	LIBLLMOD_LOG_ERROR,
   	LIBLLMOD_LOG_INFO,
   	LIBLLMOD_LOG_DEBUG,
   	LIBLLMOD_LOG_ABUSIVE
};


/* Prepare models and devices to run text generation.

   context - will return prepared context (type void*) there, should not be nullptr
   models_dir - directory holding models to load, should include "qnn_model.bin"
   log_level - logging level, can be later overwritten with libllmod_set_log_level
   use_htp - whether to use HTP or GPU

   Returns 0 if successful, otherwise an error code is returned.
   If successful, *context will be pointer to a prepare context that should be passed to other functions and cleaned when no longer needed, see release.
   Even if unsuccessful, *context might still be set if failure happened after initial object has been created, in which case it should still be released
   by a call to ``release``, it should also be used when querying for error details; it should not be, however, used to generate text.
   If a method fails before a context object is created, *context will be nullptr.
*/
LIBLLMOD_API int libllmod_setup(
	void** context, 
	const char* models_dir, 
	unsigned int log_level, 
	int use_htp,
	const char* model_type, 
	float temperature, 
	float topp, 
	unsigned long long rng_seed, 
	int max_sequence_length
);


/* Changes the log level for the provided context.

   context - a previously prepared context obtained by a call to setup
   log_level - new log level
   
   Returns 0 if successful, otherwise an error code is returned.
*/
LIBLLMOD_API int libllmod_set_log_level(void* context, unsigned int log_level);


/* Increase reference counter for a given context.

   For each additional call to ref_context, an additional call to release has to be made before
   a context will be actually cleaned.
*/
LIBLLMOD_API int libllmod_ref_context(void* context);

/* Release a previously prepared context, obtained by a call to setup.

   Returns 0 if successful, otherwise an error code is returned.
*/
LIBLLMOD_API int libllmod_release(void* context);


/* Run autoregressive text generation.

   context - a previously prepared context obtained by a call to setup
   prompt - a null-terminated, UTF8-encoded user input
   text_out - output buffer that will hold resulting text

   This function can either handle memory allocation on its own or work with a user-provided buffer.
   If the user wants to leave allocation to the function, please call it as:

    char* buffer = nullptr;
    unsigned int buffer_len = 0;

    libllmod_run(..., &buffer, &buffer_len);

   In which case the function will write to both ``buffer`` and ``buffer_len``. Note that setting initial value of ``buffer`` to ``nullptr``
   is important! Otherwise the code will assume a user-provided buffer with length 0 is to be used, which will result in an error.
   Otherwise, if user wants to reuse an existing buffer, the call should be made as:

    unsigned char* buffer = <existing_buffer>;
    unsigned int buffer_len = <length of the existing buffer>;

    generate_image(..., &buffer, &buffer_len);

   If the existing buffer is too small, the function will return an appropriate error.
   If it is too large, the function will continue its work as normal, and will return back the amount of data actually written to the buffer
   using the same ``buffer_len`` variable (similar to the case when allocation is performed by the function). Therefore it is important
   to keep the length of the original buffer as a separate variable - otherwise this information might be lost.

   In either case (allocation handled by the user, or by the function), it is the user's responsibility to free the buffer when it is no longer needed.

   Returns 0 if successful, otherwise an error code is returned.
*/
LIBLLMOD_API int libllmod_run(void* context, const char* prompt, char** text_out, int steps, int& last_token_position);


/* Return a human-readable null-terminated string describing a returned error code.

   The method can return nullptr if ``errorcode`` is not a valid error code.
*/
LIBLLMOD_API const char* libllmod_get_error_description(int errorcode);


/* Return extra information about the last error with ``errorcode`` that occurred within a given ``context``.

   In general, the information about each error is stored per-error and per-context. There are two exceptions, though,
   when it is only stored per-error:
      1. If an error occurred while setting up a context, the user should pass nullptr as ``context`` to obtain necessary information.
      2. If an error is related to calling a function with invalid context (indicated by certain error codes) then ``context`` is also ignored.

   The method can return nullptr if either ``errorcode`` is not a valid error code, if ``context`` is not a valid context,
   if ``errorcode`` has not happened for ``context`` or if no extra information has been provided by the implementation when an error was recorded.
   Otherwise a null-terminated string is returned.
*/
LIBLLMOD_API const char* libllmod_get_last_error_extra_info(int errorcode, void* context);

#ifdef __cplusplus
}
#endif
