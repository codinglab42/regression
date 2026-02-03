#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <type_traits>
#include "exceptions/io_exception.h"
#include "exceptions/validation_exception.h"
#include "exceptions/exception_macros.h"

namespace utils {

    // Interfaccia base per oggetti serializzabili
    class ISerializable {
    public:
        virtual ~ISerializable() = default;
        
        virtual void save(const std::string& filename) const = 0;
        virtual void load(const std::string& filename) = 0;
        virtual std::string to_string() const = 0;
        
        // Metodi protetti per serializzazione binaria
        virtual void serialize_binary(std::ostream& out) const = 0;
        virtual void deserialize_binary(std::istream& in) = 0;
        
        // Metodo per ottenere il tipo del modello
        virtual std::string get_model_type() const = 0;
    };

    // Template per serializzazione binaria
    template<typename Model>
    class BinarySerializer {
    public:
        static void save(const Model& model, const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                ML_THROW_IO_ERROR(filename, "open for writing", model.get_model_type());
            }
            
            try {
                // Scrive header con versione e tipo modello
                const char magic[] = "MLMOD";
                file.write(magic, sizeof(magic));
                
                uint32_t version = 1;
                file.write(reinterpret_cast<const char*>(&version), sizeof(version));
                
                // Serializza il modello
                model.serialize_binary(file);
                
                if (!file.good()) {
                    throw ml_exception::SerializationException(
                        filename, "write error", model.get_model_type());
                }
            } catch (const std::exception& e) {
                throw ml_exception::SerializationException(
                    filename, e.what(), model.get_model_type());
            }
        }
        
        static void load(Model& model, const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw ml_exception::FileNotFoundException(filename, model.get_model_type());
            }
            
            try {
                // Legge e verifica header
                char magic[6];
                file.read(magic, sizeof(magic));
                
                if (std::string(magic, sizeof(magic)) != "MLMOD") {
                    throw ml_exception::DeserializationException(
                        filename, "invalid file format", model.get_model_type());
                }
                
                uint32_t version;
                file.read(reinterpret_cast<char*>(&version), sizeof(version));
                
                if (version != 1) {
                    throw ml_exception::DeserializationException(
                        filename, "unsupported version: " + std::to_string(version),
                        model.get_model_type());
                }
                
                // Deserializza il modello
                model.deserialize_binary(file);
                
                if (!file.good() && !file.eof()) {
                    throw ml_exception::DeserializationException(
                        filename, "read error or corrupted file", model.get_model_type());
                }
            } catch (const ml_exception::MLException&) {
                throw;
            } catch (const std::exception& e) {
                throw ml_exception::DeserializationException(
                    filename, e.what(), model.get_model_type());
            }
        }
    };

    // Utility per serializzazione di tipi Eigen
    namespace eigen_utils {
        
        template<typename EigenType>
        static void serialize_eigen(const EigenType& matrix, std::ostream& out) {
            using Scalar = typename EigenType::Scalar;
            using Index = typename EigenType::Index;
            
            Index rows = matrix.rows();
            Index cols = matrix.cols();
            
            out.write(reinterpret_cast<const char*>(&rows), sizeof(Index));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(Index));
            
            if (rows > 0 && cols > 0) {
                out.write(reinterpret_cast<const char*>(matrix.data()), 
                         rows * cols * sizeof(Scalar));
            }
        }
        
        template<typename EigenType>
        static void deserialize_eigen(EigenType& matrix, std::istream& in) {
            using Scalar = typename EigenType::Scalar;
            using Index = typename EigenType::Index;
            
            Index rows, cols;
            in.read(reinterpret_cast<char*>(&rows), sizeof(Index));
            in.read(reinterpret_cast<char*>(&cols), sizeof(Index));
            
            matrix.resize(rows, cols);
            
            if (rows > 0 && cols > 0) {
                in.read(reinterpret_cast<char*>(matrix.data()), 
                       rows * cols * sizeof(Scalar));
            }
        }
        
        template<typename EigenType>
        static void serialize_eigen_vector(const EigenType& vector, std::ostream& out) {
            using Scalar = typename EigenType::Scalar;
            using Index = typename EigenType::Index;
            
            Index size = vector.size();
            out.write(reinterpret_cast<const char*>(&size), sizeof(Index));
            
            if (size > 0) {
                out.write(reinterpret_cast<const char*>(vector.data()), 
                         size * sizeof(Scalar));
            }
        }
        
        template<typename EigenType>
        static void deserialize_eigen_vector(EigenType& vector, std::istream& in) {
            using Scalar = typename EigenType::Scalar;
            using Index = typename EigenType::Index;
            
            Index size;
            in.read(reinterpret_cast<char*>(&size), sizeof(Index));
            
            vector.resize(size);
            
            if (size > 0) {
                in.read(reinterpret_cast<char*>(vector.data()), 
                       size * sizeof(Scalar));
            }
        }
    };

    // Classe base astratta per modelli serializzabili
    class SerializableModel : public ISerializable {
    public:
        // Implementazioni default che usano BinarySerializer
        void save(const std::string& filename) const override {
            BinarySerializer<std::decay_t<decltype(*this)>>::save(*this, filename);
        }
        
        void load(const std::string& filename) override {
            BinarySerializer<std::decay_t<decltype(*this)>>::load(*this, filename);
        }
        
        // Metodi che devono essere implementati dalle classi derivate
        virtual std::string to_string() const override = 0;
        virtual void serialize_binary(std::ostream& out) const override = 0;
        virtual void deserialize_binary(std::istream& in) override = 0;
        virtual std::string get_model_type() const override = 0;
    };

} // namespace serialization

#endif