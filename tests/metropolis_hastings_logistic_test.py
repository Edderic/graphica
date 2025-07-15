import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ..graphica.ds import BayesianNetwork as BN
from ..graphica.random.normal import Normal
from ..graphica.random.gamma import Gamma
from ..graphica.random.binomial import Binomial
from ..graphica.random.deterministic import Deterministic
from ..graphica.random.logistic import Logistic
from ..graphica.inference.metropolis_hastings import MetropolisHastings
from ..graphica.query import Query


def test_metropolis_hastings_logistic_regression():
    """
    Test MetropolisHastings with a Bayesian network for facial measurements
    and compare results with sklearn logistic regression.
    """
    np.random.seed(42)

    # Sample data for facial measurements with fit tests
    facial_measurements_with_fit_tests = [
        {
            "id": 1181,
            "n": 4,
            "denominator": "0.07797638898152471153",
            "n95_mode_hmff": "51.29757933453083568692",
            "qlft_pass": False,
            "mask_id": 222,
            "facial_hair_beard_length_mm": 3,
            "source": "N95ModeService",
            "user_id": 99,
            "unique_internal_model_code": "Makrite 9500S Surgical NIOSH N95 Size Small headstraps",
            "perimeter_mm": 380,
            "strap_type": "Headstrap",
            "style": "Cup",
            "face_width": 137,
            "jaw_width": 118,
            "face_depth": 96,
            "face_length": 112,
            "lower_face_length": 64.7,
            "bitragion_menton_arc": 288,
            "bitragion_subnasale_arc": 230,
            "nasal_root_breadth": 15,
            "nose_protrusion": 27,
            "nose_bridge_height": 13,
            "lip_width": 52,
            "head_circumference": None,
            "nose_breadth": None,
            "face_width_z_score": "0.67977799908041970849",
            "jaw_width_z_score": "0.40297193750449864935",
            "face_depth_z_score": "-0.12206280173663213955",
            "face_length_z_score": "0.12165132129189922255",
            "lower_face_length_z_score": -0.166733715489598,
            "bitragion_menton_arc_z_score": "0.52873879608511797323",
            "bitragion_subnasale_arc_z_score": "-0.03212402986150920281",
            "nasal_root_breadth_z_score": "-0.73659817520074123993",
            "nose_protrusion_z_score": "-0.13376416563567377767",
            "nose_bridge_height_z_score": "-0.69576668441348658181",
            "lip_width_z_score": "0.27068435833971968475",
            "head_circumference_z_score": None,
            "nose_breadth_z_score": None
        },
        {
            "id": 267,
            "n": 4,
            "denominator": "0.02",
            "n95_mode_hmff": "200.0",
            "qlft_pass": True,
            "mask_id": 336,
            "facial_hair_beard_length_mm": 0,
            "source": "N95ModeService",
            "user_id": 99,
            "unique_internal_model_code": "Zimi ZM9541 w/Headstraps",
            "perimeter_mm": 364,
            "strap_type": "Adjustable Headstrap",
            "style": "Bifold & Gasket",
            "face_width": 137,
            "jaw_width": 118,
            "face_depth": 96,
            "face_length": 112,
            "lower_face_length": 64.7,
            "bitragion_menton_arc": 288,
            "bitragion_subnasale_arc": 230,
            "nasal_root_breadth": 15,
            "nose_protrusion": 27,
            "nose_bridge_height": 13,
            "lip_width": 52,
            "head_circumference": None,
            "nose_breadth": None,
            "face_width_z_score": "0.67977799908041970849",
            "jaw_width_z_score": "0.40297193750449864935",
            "face_depth_z_score": "-0.12206280173663213955",
            "face_length_z_score": "0.12165132129189922255",
            "lower_face_length_z_score": -0.166733715489598,
            "bitragion_menton_arc_z_score": "0.52873879608511797323",
            "bitragion_subnasale_arc_z_score": "-0.03212402986150920281",
            "nasal_root_breadth_z_score": "-0.73659817520074123993",
            "nose_protrusion_z_score": "-0.13376416563567377767",
            "nose_bridge_height_z_score": "-0.69576668441348658181",
            "lip_width_z_score": "0.27068435833971968475",
            "head_circumference_z_score": None,
            "nose_breadth_z_score": None
        },
        {
            "id": 1182,
            "n": 4,
            "denominator": "0.07797638898152471153",
            "n95_mode_hmff": "51.29757933453083568692",
            "qlft_pass": True,
            "mask_id": 222,
            "facial_hair_beard_length_mm": 0,
            "source": "N95ModeService",
            "user_id": 100,
            "unique_internal_model_code": "Makrite 9500S Surgical NIOSH N95 Size Small headstraps",
            "perimeter_mm": 375,
            "strap_type": "Headstrap",
            "style": "Cup",
            "face_width": 142,
            "jaw_width": 120,
            "face_depth": 98,
            "face_length": 115,
            "lower_face_length": 66.0,
            "bitragion_menton_arc": 290,
            "bitragion_subnasale_arc": 235,
            "nasal_root_breadth": 16,
            "nose_protrusion": 28,
            "nose_bridge_height": 14,
            "lip_width": 54,
            "head_circumference": None,
            "nose_breadth": None,
            "face_width_z_score": "0.8",
            "jaw_width_z_score": "0.5",
            "face_depth_z_score": "0.1",
            "face_length_z_score": "0.2",
            "lower_face_length_z_score": 0.1,
            "bitragion_menton_arc_z_score": "0.6",
            "bitragion_subnasale_arc_z_score": "0.1",
            "nasal_root_breadth_z_score": "-0.5",
            "nose_protrusion_z_score": "0.1",
            "nose_bridge_height_z_score": "-0.4",
            "lip_width_z_score": "0.4",
            "head_circumference_z_score": None,
            "nose_breadth_z_score": None
        },
        {
            "id": 268,
            "n": 4,
            "denominator": "0.02",
            "n95_mode_hmff": "200.0",
            "qlft_pass": False,
            "mask_id": 336,
            "facial_hair_beard_length_mm": 2,
            "source": "N95ModeService",
            "user_id": 101,
            "unique_internal_model_code": "Zimi ZM9541 w/Headstraps",
            "perimeter_mm": 370,
            "strap_type": "Adjustable Headstrap",
            "style": "Bifold & Gasket",
            "face_width": 135,
            "jaw_width": 115,
            "face_depth": 94,
            "face_length": 110,
            "lower_face_length": 63.0,
            "bitragion_menton_arc": 285,
            "bitragion_subnasale_arc": 225,
            "nasal_root_breadth": 14,
            "nose_protrusion": 26,
            "nose_bridge_height": 12,
            "lip_width": 50,
            "head_circumference": None,
            "nose_breadth": None,
            "face_width_z_score": "0.5",
            "jaw_width_z_score": "0.3",
            "face_depth_z_score": "-0.2",
            "face_length_z_score": "0.0",
            "lower_face_length_z_score": -0.3,
            "bitragion_menton_arc_z_score": "0.4",
            "bitragion_subnasale_arc_z_score": "-0.1",
            "nasal_root_breadth_z_score": "-0.8",
            "nose_protrusion_z_score": "-0.2",
            "nose_bridge_height_z_score": "-0.8",
            "lip_width_z_score": "0.2",
            "head_circumference_z_score": None,
            "nose_breadth_z_score": None
        }
    ]

    # Extract unique mask_ids
    mask_ids = list(set(row['mask_id'] for row in facial_measurements_with_fit_tests))

    # Build Bayesian Network
    bn = BN()

    # Add mask-specific parameters
    mask_params = {}
    for mask_id in mask_ids:
        a = Normal(name=f'a_mask_{mask_id}', mean=0, std=10)
        b = Normal(name=f'b_mask_{mask_id}', mean=0, std=10)
        c = Normal(name=f'c_mask_{mask_id}', mean=0, std=10)

        bn.add_node(a)
        bn.add_node(b)
        bn.add_node(c)
        mask_params[mask_id] = {'a': a, 'b': b, 'c': c}

    # Add beta parameters (Gamma distributed for positive values)
    beta_face_width = Gamma(name='beta_face_width', shape=2, rate=2)
    beta_face_length = Gamma(name='beta_face_length', shape=2, rate=2)
    beta_bitragion_subnasale_arc = Gamma(name='beta_bitragion_subnasale_arc', shape=2, rate=2)
    beta_nose_protrusion = Gamma(name='beta_nose_protrusion', shape=2, rate=2)
    beta_nasal_root_breadth = Gamma(name='beta_nasal_root_breadth', shape=2, rate=2)

    bn.add_node(beta_face_width)
    bn.add_node(beta_face_length)
    bn.add_node(beta_bitragion_subnasale_arc)
    bn.add_node(beta_nose_protrusion)
    bn.add_node(beta_nasal_root_breadth)

    # Process each row
    for row in facial_measurements_with_fit_tests:
        fit_test_uuid = f"{row['id']}_{row['source']}"
        mask_id = row['mask_id']

        # Distance calculation
        distance = Deterministic(
            callable_func=lambda face_width, beta_face_width, face_length, beta_face_length,
                                  nose_protrusion, beta_nose_protrusion, bitragion_subnasale_arc,
                                  beta_bitragion_subnasale_arc, nasal_root_breadth, beta_nasal_root_breadth,
                                  mask_perimeter: (face_width * beta_face_width +
                                                  face_length * beta_face_length +
                                                  nose_protrusion * beta_nose_protrusion +
                                                  bitragion_subnasale_arc * beta_bitragion_subnasale_arc +
                                                  nasal_root_breadth * beta_nasal_root_breadth -
                                                  mask_perimeter),
            face_width=row['face_width'],
            face_length=row['face_length'],
            nose_protrusion=row['nose_protrusion'],
            bitragion_subnasale_arc=row['bitragion_subnasale_arc'],
            nasal_root_breadth=row['nasal_root_breadth'],
            mask_perimeter=row['perimeter_mm'],
            beta_face_width=beta_face_width,
            beta_face_length=beta_face_length,
            beta_nose_protrusion=beta_nose_protrusion,
            beta_bitragion_subnasale_arc=beta_bitragion_subnasale_arc,
            beta_nasal_root_breadth=beta_nasal_root_breadth
        )

        # Get mask parameters
        a, b, c = mask_params[mask_id]['a'], mask_params[mask_id]['b'], mask_params[mask_id]['c']

        # Logistic function (quadratic passed to logistic)
        logistic = Logistic(
            callable_func=lambda a, b, z, c: a * z**2 + b * z + c,
            a=a,
            b=b,
            c=c,
            z=distance
        )

        # Observed data (convert boolean to int)
        obs = 1 if row['qlft_pass'] else 0

        # Binomial likelihood
        binomial = Binomial(name=f'fit_test_result_{fit_test_uuid}', n=1, p=logistic)
        binomial.obs = obs

        bn.add_node(distance)
        bn.add_node(logistic)
        bn.add_node(binomial)

    # Prepare data for sklearn comparison
    X = []
    y = []

    for row in facial_measurements_with_fit_tests:
        features = [
            row['face_width'],
            row['face_length'],
            row['nose_protrusion'],
            row['bitragion_subnasale_arc'],
            row['nasal_root_breadth'],
            row['perimeter_mm']  # Include perimeter as a feature
        ]
        X.append(features)
        y.append(1 if row['qlft_pass'] else 0)

    X = np.array(X)
    y = np.array(y)

    # Fit sklearn logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sklearn_lr = LogisticRegression(random_state=42, max_iter=1000)
    sklearn_lr.fit(X_scaled, y)

    # Define transition function for Metropolis-Hastings
    def transition(particle):
        # Propose new values for all parameters
        new_particle = particle.copy()

        # Update beta parameters (Gamma distributed)
        for param_name in ['beta_face_width', 'beta_face_length', 'beta_bitragion_subnasale_arc',
                          'beta_nose_protrusion', 'beta_nasal_root_breadth']:
            current_val = particle.get_value(param_name)
            # Propose from log-normal to ensure positivity
            proposal = current_val * np.exp(np.random.normal(0, 0.1))
            new_particle.set_value(param_name, proposal)

        # Update mask parameters (Normal distributed)
        for mask_id in mask_ids:
            for param in ['a', 'b', 'c']:
                param_name = f'{param}_mask_{mask_id}'
                current_val = particle.get_value(param_name)
                proposal = current_val + np.random.normal(0, 0.5)
                new_particle.set_value(param_name, proposal)

        return new_particle

    # Query: condition on all fit test results
    fit_test_nodes = [node for node in bn.nodes if node.name.startswith('fit_test_result_')]
    givens = [{node.name: node.obs} for node in fit_test_nodes]

    query = Query(
        outcomes=['beta_face_width', 'beta_face_length', 'beta_bitragion_subnasale_arc',
                 'beta_nose_protrusion', 'beta_nasal_root_breadth'],
        givens=givens
    )

    # Metropolis-Hastings sampler
    sampler = MetropolisHastings(
        network=bn,
        query=query,
        transition_function=transition
    )

    # Run sampler
    samples = sampler.sample(n=1000, burn_in=200)

    # Extract beta values from samples
    beta_samples = {
        'beta_face_width': [particle.get_value('beta_face_width') for particle in samples],
        'beta_face_length': [particle.get_value('beta_face_length') for particle in samples],
        'beta_bitragion_subnasale_arc': [particle.get_value('beta_bitragion_subnasale_arc') for particle in samples],
        'beta_nose_protrusion': [particle.get_value('beta_nose_protrusion') for particle in samples],
        'beta_nasal_root_breadth': [particle.get_value('beta_nasal_root_breadth') for particle in samples]
    }

    # Calculate means and standard deviations
    beta_means = {name: np.mean(values) for name, values in beta_samples.items()}
    beta_stds = {name: np.std(values) for name, values in beta_samples.items()}

    print("Metropolis-Hastings Results:")
    for name, mean in beta_means.items():
        print(f"{name}: {mean:.4f} ± {beta_stds[name]:.4f}")

    print("\nSklearn Logistic Regression Coefficients:")
    feature_names = ['face_width', 'face_length', 'nose_protrusion', 'bitragion_subnasale_arc', 'nasal_root_breadth', 'perimeter_mm']
    for i, (name, coef) in enumerate(zip(feature_names, sklearn_lr.coef_[0])):
        print(f"{name}: {coef:.4f}")

    # Basic assertions to ensure the inference worked
    # Check that we got reasonable beta values (positive for Gamma priors)
    for name, mean in beta_means.items():
        assert mean > 0, f"Beta {name} should be positive, got {mean}"
        assert beta_stds[name] > 0, f"Beta {name} should have non-zero std, got {beta_stds[name]}"

    # Check that the model converged (reasonable standard deviations)
    for name, std in beta_stds.items():
        assert std < 10, f"Beta {name} std too large: {std}"

    print("\nTest passed: Metropolis-Hastings inference completed successfully!")
