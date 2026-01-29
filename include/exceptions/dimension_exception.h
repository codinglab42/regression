#ifndef DIMENSION_EXCEPTION_H
#define DIMENSION_EXCEPTION_H

#include "ml_exception.h"

namespace ml_exception {

    class DimensionMismatchException : public MLException {
    public:
        DimensionMismatchException(const std::string& operation,
                                 int expected_rows, int expected_cols,
                                 int actual_rows, int actual_cols,
                                 const std::string& model_type = "")
            : MLException(build_message(operation, expected_rows, expected_cols,
                                      actual_rows, actual_cols), 
                        model_type) {}
    
    private:
        static std::string build_message(const std::string& operation,
                                       int er, int ec, int ar, int ac) {
            std::ostringstream oss;
            oss << "Dimension mismatch in " << operation 
                << ": expected (" << er << ", " << ec 
                << "), got (" << ar << ", " << ac << ")";
            return oss.str();
        }
    };

    class FeatureMismatchException : public MLException {
    public:
        FeatureMismatchException(int expected_features, int actual_features,
                               const std::string& model_type = "")
            : MLException(build_message(expected_features, actual_features),
                        model_type) {}
    
    private:
        static std::string build_message(int expected, int actual) {
            std::ostringstream oss;
            oss << "Feature mismatch: expected " << expected 
                << " features, got " << actual;
            return oss.str();
        }
    };

    class LabelMismatchException : public MLException {
    public:
        LabelMismatchException(int expected_classes, int actual_classes,
                             const std::string& model_type = "")
            : MLException(build_message(expected_classes, actual_classes),
                        model_type) {}
    
    private:
        static std::string build_message(int expected, int actual) {
            std::ostringstream oss;
            oss << "Label mismatch: expected " << expected 
                << " unique classes, got " << actual;
            return oss.str();
        }
    };

} // namespace ml_exception

#endif