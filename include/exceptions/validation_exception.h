#ifndef VALIDATION_EXCEPTION_H
#define VALIDATION_EXCEPTION_H

#include "ml_exception.h"

namespace ml_exception {

    class ValidationException : public MLException {
    public:
        explicit ValidationException(const std::string& error_msg,
                                  const std::string& model_type = "")
            : MLException("Validation error: " + error_msg, model_type) {}
    };

    class InvalidConfigurationException : public MLException {
    public:
        InvalidConfigurationException(const std::string& config_error,
                                    const std::string& model_type = "")
            : MLException("Invalid configuration: " + config_error, model_type) {}
    };

} // namespace ml_exception

#endif