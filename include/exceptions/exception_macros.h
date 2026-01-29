#ifndef EXCEPTION_MACROS_H
#define EXCEPTION_MACROS_H

#include "dimension_exception.h"
#include "fitting_exception.h"
#include "validation_exception.h"

// Macro per controlli rapidi
#define ML_CHECK_FITTED(condition, model_type) \
    do { \
        if (!(condition)) \
            throw ml_exception::NotFittedException(model_type); \
    } while(0)

#define ML_CHECK_DIMENSIONS(actual_rows, expected_rows, \
                           actual_cols, expected_cols, \
                           operation, model_type) \
    do { \
        if ((actual_rows) != (expected_rows) || \
            (actual_cols) != (expected_cols)) \
            throw ml_exception::DimensionMismatchException( \
                operation, expected_rows, expected_cols, \
                actual_rows, actual_cols, model_type); \
    } while(0)

#define ML_CHECK_FEATURES(actual_features, expected_features, model_type) \
    do { \
        if ((actual_features) != (expected_features)) \
            throw ml_exception::FeatureMismatchException( \
                expected_features, actual_features, model_type); \
    } while(0)

#define ML_CHECK_PARAM(condition, param_name, requirement, model_type) \
    do { \
        if (!(condition)) \
            throw ml_exception::InvalidParameterException( \
                param_name, requirement, model_type); \
    } while(0)

#define ML_CHECK_NOT_EMPTY(data, data_name, model_type) \
    do { \
        if ((data).rows() == 0 || (data).cols() == 0) \
            throw ml_exception::EmptyDatasetException(data_name, model_type); \
    } while(0)

#define ML_THROW_IO_ERROR(filename, operation, model_type) \
    throw ml_exception::IOException(filename, operation, model_type)

#endif