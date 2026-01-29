#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>
#include <memory>
#include <string>

namespace activation {

    class Activation {
    public:
        virtual ~Activation() = default;
        
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& z) = 0;
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& dA, 
                                       const Eigen::MatrixXd& z) = 0;
        
        virtual std::string get_type() const = 0;
        virtual Activation* clone() const = 0;
    };

    // Factory per creare funzioni di attivazione
    std::unique_ptr<Activation> create_activation(const std::string& type);
    
    // Funzioni disponibili
    class ReLU : public Activation {
    public:
        Eigen::MatrixXd forward(const Eigen::MatrixXd& z) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& dA, 
                               const Eigen::MatrixXd& z) override;
        std::string get_type() const override { return "relu"; }
        ReLU* clone() const override { return new ReLU(*this); }
    };
    
    class Sigmoid : public Activation {
    public:
        Eigen::MatrixXd forward(const Eigen::MatrixXd& z) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& dA, 
                               const Eigen::MatrixXd& z) override;
        std::string get_type() const override { return "sigmoid"; }
        Sigmoid* clone() const override { return new Sigmoid(*this); }
    };
    
    class Tanh : public Activation {
    public:
        Eigen::MatrixXd forward(const Eigen::MatrixXd& z) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& dA, 
                               const Eigen::MatrixXd& z) override;
        std::string get_type() const override { return "tanh"; }
        Tanh* clone() const override { return new Tanh(*this); }
    };
    
    class Softmax : public Activation {
    public:
        Eigen::MatrixXd forward(const Eigen::MatrixXd& z) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& dA, 
                               const Eigen::MatrixXd& z) override;
        std::string get_type() const override { return "softmax"; }
        Softmax* clone() const override { return new Softmax(*this); }
    };
    
    class LeakyReLU : public Activation {
    private:
        double alpha_;
    public:
        explicit LeakyReLU(double alpha = 0.01) : alpha_(alpha) {}
        Eigen::MatrixXd forward(const Eigen::MatrixXd& z) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& dA, 
                               const Eigen::MatrixXd& z) override;
        std::string get_type() const override { return "leaky_relu"; }
        LeakyReLU* clone() const override { return new LeakyReLU(*this); }
    };

} // namespace activation

#endif