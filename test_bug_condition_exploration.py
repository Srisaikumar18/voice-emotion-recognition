"""
Bug Condition Exploration Test for Emotion Model Compatibility Fix

This test explores the bug condition where a model trained with an older version
of scikit-learn (1.3.0) fails when loaded in a newer version (1.7.2) due to the
missing 'monotonic_cst' attribute.

**CRITICAL**: This test is EXPECTED TO FAIL on unfixed code.
The failure confirms that the bug exists and demonstrates the AttributeError.

**Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.3, 2.4**
"""

import os
import joblib
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck


# Property 1: Fault Condition - Model Version Incompatibility
@given(
    # Generate realistic 40-dimensional MFCC feature vectors
    # MFCC features are typically in the range [-50, 50] with most values near 0
    features=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=40,
        max_size=40
    )
)
@settings(
    max_examples=3,  # Run 3 test cases to explore the input space (reduced for faster execution)
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_model_predict_without_attribute_error(features):
    """
    Property: Model should predict emotions without AttributeError
    
    For any valid 40-dimensional MFCC feature vector, the model should:
    1. Load successfully from emotion_model.pkl
    2. Accept the feature vector for prediction
    3. Return a valid emotion label from {"Happy", "Sad", "Angry", "Neutral"}
    4. NOT raise AttributeError related to 'monotonic_cst'
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: 
    This test will FAIL with AttributeError: 'DecisionTreeClassifier' object 
    has no attribute 'monotonic_cst'
    
    This failure is CORRECT - it proves the bug exists.
    """
    # Load the existing model (trained with scikit-learn 1.3.0)
    model_path = os.path.join("model", "emotion_model.pkl")
    
    # Verify model file exists
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    # Load model (this should succeed even with version mismatch)
    model = joblib.load(model_path)
    
    # Convert features to numpy array
    feature_array = np.array(features).reshape(1, -1)
    
    # Attempt prediction - this is where the bug manifests
    # On unfixed code, this will raise AttributeError
    prediction = model.predict(feature_array)
    
    # Verify prediction is a valid emotion label
    valid_emotions = {"Happy", "Sad", "Angry", "Neutral"}
    assert len(prediction) == 1, "Prediction should return exactly one result"
    assert prediction[0] in valid_emotions, f"Prediction '{prediction[0]}' is not a valid emotion"
    
    # If we reach here, the bug is fixed (or doesn't exist in this environment)
    print(f"✓ Prediction successful: {prediction[0]} for features sample: {features[:3]}...")


if __name__ == "__main__":
    # Run the property-based test
    print("=" * 70)
    print("Bug Condition Exploration Test")
    print("=" * 70)
    print("\nThis test explores the model version incompatibility bug.")
    print("EXPECTED: Test FAILS with AttributeError on unfixed code")
    print("This failure confirms the bug exists.\n")
    
    try:
        test_model_predict_without_attribute_error()
        print("\n" + "=" * 70)
        print("UNEXPECTED: Test PASSED")
        print("=" * 70)
        print("\nThe test passed, which means either:")
        print("1. The bug has already been fixed")
        print("2. The model is compatible with the current scikit-learn version")
        print("3. The root cause analysis may need revision")
    except AttributeError as e:
        if "monotonic_cst" in str(e):
            print("\n" + "=" * 70)
            print("EXPECTED: Test FAILED with AttributeError")
            print("=" * 70)
            print(f"\nCounterexample found: {e}")
            print("\nThis confirms the bug exists:")
            print("- Model trained with scikit-learn 1.3.0")
            print("- Runtime environment has scikit-learn 1.7.2")
            print("- Missing 'monotonic_cst' attribute causes prediction to fail")
            print("\nBug condition successfully demonstrated!")
        else:
            print(f"\nUnexpected AttributeError: {e}")
            raise
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        raise
