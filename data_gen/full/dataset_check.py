import numpy as np
import pytest

file_names = ['QCD_100k.npz','HToBB_100k.npz','HToCC_100k.npz','HToGG_100k.npz','HToWW4Q_100k.npz','ZToQQ_100k.npz','WToQQ_100k.npz','TTBar_100k.npz']
list_of_x = []
list_of_y = []

expected_x_length = 100000
expected_y_length = 100000

expected_y_range = np.arange(10)

for file_name in file_names:
    file = np.load(file_name)
    x = file['x']
    y = file['y']
    
    list_of_x.append(x)
    list_of_y.append(y)

test_for_x_length = [(list_of_x[i], expected_x_length) for i in range(len(file_names))]
test_for_y_length = [(list_of_y[i], expected_y_length) for i in range(len(file_names))]
test_for_y_range = [(list_of_y[i], expected_y_range) for i in range(len(file_names))]

@pytest.mark.parametrize("x, expected_length", test_for_x_length)
def test_x_length(x: np.ndarray, expected_length: int):
    assert x.shape[0] == expected_length, f"Expected {expected_length} jets, got {x.shape[0]}"

@pytest.mark.parametrize("y, expected_length", test_for_y_length)
def test_y_length(y: np.ndarray, expected_length: int):
    assert y.shape[0] == expected_length, f"Expected {expected_length} labels, got {y.shape[0]}"
    
@pytest.mark.parametrize("y, expected_range", test_for_y_range)
def test_y_range(y: np.ndarray, expected_range: np.ndarray):
    unique_values = np.unique(y)
    assert np.all(np.isin(unique_values, expected_range)), f"Unexpected label(s) in y: {unique_values[~np.isin(unique_values, expected_range)]}"

test_for_pt = [list_of_x[i][:,:,0] for i in range(len(file_names))]
@pytest.mark.parametrize("pt", test_for_pt)
def test_pt_nonnegative(pt: np.ndarray):
    assert np.all(pt >= 0), "Found negative pt values"

test_for_phi = [list_of_x[i][:,:,2] for i in range(len(file_names))]
@pytest.mark.parametrize("phi", test_for_phi)
def test_phi_within_PI(phi: np.ndarray):
    for i, event_phi in enumerate(phi):
        if event_phi[event_phi!=0].size == 0:
            print(f"Event {i} has all phi values as zero, skipping spread test.")
            continue
        min_nonzero = event_phi[event_phi != 0].min()
        max_nonzero = event_phi[event_phi != 0].max()
        spread = max_nonzero - min_nonzero
        assert spread <= np.pi, (
            fr"TEST FAILED: \phi spread exceeds \pi. Spread = {spread}, "
            fr"min = {min_nonzero}, max = {max_nonzero}, index = {i}"
        )

test_for_mass = [list_of_x[i][:,:,3] for i in range(len(file_names))]
@pytest.mark.parametrize("mass", test_for_mass)
def test_mass_nonnegative(mass: np.ndarray):
    # print(mass[:5])
    assert np.all(mass >= 0), "Found negative mass values"

# def test_x_length(self, x: np.ndarray, length: int) -> None:
#     if x.shape[0] != length:
#         print("TEST FAILED: There should be %d jets in the dataset", length)
#     else:
#         print("TEST PASSED")
        
# def test_y_length(self, y: np.ndarray, length: int) -> None:
#     if y.shape[0] != length:
#         print("TEST FAILED: There should be %d labels for the dataset", length)
#     else:
#         print("TEST PASSED")
        
# def test_pt(self, pt:np.ndarray) -> None:
#     if np.all(pt >= 0):
#         print("TEST PASSED")