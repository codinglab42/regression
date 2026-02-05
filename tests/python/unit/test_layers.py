import pytest
import machine_learning_module as ml

def test_dense_layer_dimensions():
    """Verifica che il layer Dense mantenga la coerenza delle dimensioni."""
    input_sz = 10
    output_sz = 5
    layer = ml.Dense(input_sz, output_sz, "relu", "he")
    
    assert layer.get_input_size() == input_sz
    assert layer.get_output_size() == output_sz
    # Parametri = (10 * 5) pesi + 5 bias = 55
    assert layer.get_parameter_count() == 55

def test_layer_summary_output(capsys):
    """Verifica che il metodo summary stampi correttamente."""
    nn = ml.NeuralNetwork()
    nn.add_layer(ml.Dense(2, 4, "relu", "he"))
    nn.summary()
    
    captured = capsys.readouterr()
    assert "Dense" in captured.out
    assert "Total params: 12" in captured.out # (2*4)+4