#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <Eigen/Dense>
#include <string>
#include "serialization/serializable.h"

namespace regression {

    class Estimator : public serialization::SerializableModel {
    public:
        virtual ~Estimator() = default;

        // Metodi virtuali puri che ogni modello deve implementare
        virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
        virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;
        virtual double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const = 0;
        
        // Metodi ereditati da SerializableModel
        std::string to_string() const override = 0;
        void serialize_binary(std::ostream& out) const override = 0;
        void deserialize_binary(std::istream& in) override = 0;
        std::string get_model_type() const override = 0;
        
        // Manteniamo save/load per compatibilità (già implementati in SerializableModel)
        using serialization::SerializableModel::save;
        using serialization::SerializableModel::load;
    };

} // namespace regression

#endif