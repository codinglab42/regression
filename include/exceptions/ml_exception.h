#ifndef ML_EXCEPTION_H
#define ML_EXCEPTION_H

#include <stdexcept>
#include <string>
#include <sstream>
#include <filesystem>

namespace ml_exception {

    class MLException : public std::runtime_error {
    public:
        explicit MLException(const std::string& msg, 
                           const std::string& model_type = "")
            : std::runtime_error(build_message(msg, model_type)),
              model_type_(model_type) {}
        
        const std::string& get_model_type() const { return model_type_; }
    
    protected:
        std::string model_type_;
        
        static std::string build_message(const std::string& msg, 
                                       const std::string& model_type) {
            std::ostringstream oss;
            if (!model_type.empty()) {
                oss << "[" << model_type << "] ";
            }
            oss << msg;
            return oss.str();
        }
    };

} // namespace ml_exception

#endif