#ifndef IO_EXCEPTION_H
#define IO_EXCEPTION_H

#include "ml_exception.h"

namespace ml_exception {

    class IOException : public MLException {
    public:
        IOException(const std::string& filename, const std::string& operation,
                  const std::string& model_type = "")
            : MLException(build_message(filename, operation), model_type) {}
    
    private:
        static std::string build_message(const std::string& filename,
                                       const std::string& operation) {
            return "IO error during " + operation + 
                   " for file: " + filename;
        }
    };

    class FileNotFoundException : public IOException {
    public:
        explicit FileNotFoundException(const std::string& filename,
                                    const std::string& model_type = "")
            : IOException(filename, "file not found", model_type) {}
    };

    class SerializationException : public IOException {
    public:
        SerializationException(const std::string& filename,
                             const std::string& error_detail,
                             const std::string& model_type = "")
            : IOException(build_message(filename, error_detail), model_type) {}
    
    private:
        static std::string build_message(const std::string& filename,
                                       const std::string& detail) {
            return "Serialization error for file '" + filename + 
                   "': " + detail;
        }
    };

    class DeserializationException : public IOException {
    public:
        DeserializationException(const std::string& filename,
                               const std::string& error_detail,
                               const std::string& model_type = "")
            : IOException(build_message(filename, error_detail), model_type) {}
    
    private:
        static std::string build_message(const std::string& filename,
                                       const std::string& detail) {
            return "Deserialization error for file '" + filename + 
                   "': " + detail;
        }
    };

} // namespace ml_exception

#endif