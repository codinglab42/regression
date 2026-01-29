#ifndef FITTING_EXCEPTION_H
#define FITTING_EXCEPTION_H

#include "ml_exception.h"

namespace ml_exception {

    class NotFittedException : public MLException {
    public:
        explicit NotFittedException(const std::string& model_type = "")
            : MLException("Model must be fitted before calling predict()", 
                        model_type) {}
    };

    class InvalidParameterException : public MLException {
    public:
        InvalidParameterException(const std::string& param_name,
                                const std::string& requirement,
                                const std::string& model_type = "")
            : MLException(build_message(param_name, requirement), model_type) {}
    
    private:
        static std::string build_message(const std::string& name, 
                                       const std::string& req) {
            return "Invalid parameter '" + name + "': " + req;
        }
    };

    class ConvergenceException : public MLException {
    public:
        explicit ConvergenceException(int max_iter, 
                                   const std::string& model_type = "")
            : MLException(build_message(max_iter), model_type) {}
    
    private:
        static std::string build_message(int max_iter) {
            return "Model did not converge after " + 
                   std::to_string(max_iter) + " iterations";
        }
    };

    class EmptyDatasetException : public MLException {
    public:
        explicit EmptyDatasetException(const std::string& dataset_name,
                                     const std::string& model_type = "")
            : MLException(build_message(dataset_name), model_type) {}
    
    private:
        static std::string build_message(const std::string& dataset) {
            return "Empty dataset: " + dataset;
        }
    };

} // namespace ml_exception

#endif