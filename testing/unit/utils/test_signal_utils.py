from signal_utils import time_warp_array
import torch

#pytest that checks:
#standard warp factor, stretch, compress, extreme values, small tensor, large tensor, zero time steps
#add more tests to ensure robustness

base_atol, base_rtol = 0.01, 0.01  # Base tolerances

def test_time_warp_array_standard():
    
    T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (0.5, 1.5)
    warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)
    print("warp_factor", global_time_factor.item())
    
    #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.15 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.15 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimesion check
    assert (warped_tensor.shape[1] == torch.floor(mock_tensor.shape[1] * global_time_factor).int())
    
    #mean and variance preservance check
    #please excuse the print statements; they are for debugging purposes
    
    print(warped_tensor[:, :, 0, 0, 0, 0].var(), mock_tensor[:, :, 0, 0, 0, 0].var())
    print(warped_tensor[:, :, 0, 0, 0, 0].mean(), mock_tensor[:, :, 0, 0, 0, 0].mean())
        
    print("difference: ", torch.abs(warped_tensor[:, :, 0, 0, 0, 0].mean() - mock_tensor[:, :, 0, 0, 0, 0].mean()))
    print("tolerance: ", atol + rtol * torch.abs(mock_tensor[:, :, 0, 0, 0, 0].mean()))
    print(torch.allclose(warped_tensor[:, :, 0, 0, 0, 0].mean(), mock_tensor[:, :, 0, 0, 0, 0].mean(), atol=atol, rtol=rtol))


    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 0].mean(), mock_tensor[:, :, 0, 0, 0, 0].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 0].var(), mock_tensor[:, :, 0, 0, 0, 0].var(), atol = 0.02, rtol=rtol)

    print(warped_tensor[:, :, 0, 0, 0, 1].var(), mock_tensor[:, :, 0, 0, 0, 1].var())
    print(warped_tensor[:, :, 0, 0, 0, 1].mean(), mock_tensor[:, :, 0, 0, 0, 1].mean())
    
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)


    print(warped_tensor[:, :, 0, 0, 0, 2].var(), mock_tensor[:, :, 0, 0, 0, 2].var())
    print(warped_tensor[:, :, 0, 0, 0, 2].mean(), mock_tensor[:, :, 0, 0, 0, 2].mean())

    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 2:].mean(), mock_tensor[:, :, 0, 0, 0, 2:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 2:].var(), mock_tensor[:, :, 0, 0, 0, 2:].var(), atol=atol, rtol=rtol)


def test_time_warp_array_stretch():
    T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (1.5, 2)
    warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)

        #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.1 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.1 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimension check
    assert (warped_tensor.shape[1] == torch.floor(mock_tensor.shape[1] * global_time_factor).int())

    #mean and variance preservance check
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)

def test_time_warp_array_compress():
    T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (1.2, 1.5)
    warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)
        
    #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.1 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.1 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimension check
    assert (warped_tensor.shape[1] == torch.floor(mock_tensor.shape[1] * global_time_factor).int())

    #mean and variance preservance check
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)


def test_time_warp_array_extremes():
    T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (0.1, 10.0)
    warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)

        
    #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.1 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.1 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimension check
    assert (warped_tensor.shape[1] == torch.floor(mock_tensor.shape[1] * global_time_factor).int())

    #mean and variance preservance check
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)

def test_time_warp_array_small_tensor():
    T, A, C, H, W = 2, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (0.5, 1.5)
    warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)
        
    #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.1 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.1 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimension check
    assert (warped_tensor.shape[1] == torch.floor(mock_tensor.shape[1] * global_time_factor).int())

    #mean and variance preservance check
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)

def test_time_warp_array_large_tensor():
    T, A, C, H, W = 10000, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (0.5, 1.5)
    warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)
        
    #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.1 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.1 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimension check
    assert (warped_tensor.shape[1] == torch.floor(mock_tensor.shape[1] * global_time_factor).int())

    #mean and variance preservance check
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)

def test_time_warp_array_zero_time_steps():
    T, A, C, H, W = 0, 5, 3, 4, 4  # Adjust these dimensions as needed
    mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

    warp_factor_range = (0.5, 1.5)
    try:
        warped_tensor, global_time_factor = time_warp_array(mock_tensor, warp_factor_range)
    except ValueError as e:
        assert isinstance(e, ValueError)
        return True
        
    #dynamic tolerance based on global_time_factor
    tolerance_scale = abs(1 - global_time_factor.item())

    atol =  0.1 + base_atol + base_atol * tolerance_scale 
    print("atol: ", atol)
    rtol = 0.1 + base_rtol + base_rtol * tolerance_scale
    print("rtol: ", rtol)
    
    #dimension check
    assert (warped_tensor.shape[1] == 0)
    
    #mean and variance preservance check
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].mean(), mock_tensor[:, :, 0, 0, 0, 1:].mean(), atol=atol, rtol=rtol)
    assert torch.allclose(warped_tensor[:, :, 0, 0, 0, 1:].var(), mock_tensor[:, :, 0, 0, 0, 1:].var(), atol=atol, rtol=rtol)

# def test_time_warp_array_non_standard_shape():
#     T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
#     mock_tensor = torch.rand(2, T, A, C, H, W)  # Creates a six-dim tensor with random values

#     warp_factor_range = (0.5, 1.5)
#     try:
#         warped_tensor = time_warp_array(mock_tensor, warp_factor_range)
#     except Exception as e:
#         assert isinstance(e, ValueError)

# def test_time_warp_array_negative_warp_factor_range():
#     T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
#     mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

#     warp_factor_range = (-0.5, 0.5)
#     try:
#         warped_tensor = time_warp_array(mock_tensor, warp_factor_range)
#     except Exception as e:
#         assert isinstance(e, ValueError)

# def test_time_warp_array_identical_warp_factors():
#     T, A, C, H, W = 10, 5, 3, 4, 4  # Adjust these dimensions as needed
#     mock_tensor = torch.rand(1, T, A, C, H, W)  # Creates a six-dim tensor with random values

#     warp_factor_range = (1.0, 1.0)
#     warped_tensor = time_warp_array(mock_tensor, warp_factor_range)

#     assert (warped_tensor == mock_tensor).all()
    
    
