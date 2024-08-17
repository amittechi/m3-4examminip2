import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from adultcensus_model.config.core import config
from adultcensus_model.processing.features import CustomMapper
from adultcensus_model.processing.features import ModeImputer
#from adultcensus_model.processing.features import CustomOrdinalEncoder
#from adultcensus_model.processing.features import OutlierHandler


adultcensus_pipe = Pipeline(
    steps=[
        ('modeimputation', ModeImputer(variables=config.model_config.modeimputer_fields)),
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('map_cat_features',
                    Pipeline(steps=[
                        ('map_sex', CustomMapper(config.model_config.sex_var, config.model_config.sex_mappings)),
                        ('map_workclass', CustomMapper(config.model_config.workclass_var, config.model_config.workclass_mappings)),
                        ('map_education', CustomMapper(config.model_config.education_var, config.model_config.education_mappings)),
                        ('map_marital_status', CustomMapper(config.model_config.marital_status_var, config.model_config.marital_status_mappings)),
                        ('map_occupation', CustomMapper(config.model_config.occupation_var, config.model_config.occupation_mappings)),
                        ('map_relationship', CustomMapper(config.model_config.relationship_var, config.model_config.relationship_mappings)),
                        ('map_race', CustomMapper(config.model_config.race_var, config.model_config.race_mappings)),
                        ('map_native_country', CustomMapper(config.model_config.native_country_var, config.model_config.native_country_mappings)),
                        ('cat_scaler', StandardScaler())
                    ]), config.model_config.cat_features
                ),
                ('num_feature_scaler', StandardScaler(), config.model_config.num_features)
            ],
        )),
        ('model_rf', RandomForestClassifier(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth, 
                                            random_state=config.model_config.random_state))
    ]
)