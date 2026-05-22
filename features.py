FEATURES_NUM = [
    'accommodates', 'bedrooms', 'bathrooms', 'minimum_nights',
    'host_listings_count', 'host_acceptance_rate', 'host_response_rate',
    'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'instant_bookable', 'availability_365',
    'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
    'review_scores_value', 'review_scores_accuracy', 'review_scores_checkin',
    'review_scores_communication',
    'number_of_reviews', 'number_of_reviews_ltm',
    'latitude', 'longitude',
    'calculated_host_listings_count',
    'estimated_occupancy_l365d',
    'has_wifi', 'has_tv', 'has_kitchen', 'has_ac', 'has_elevator',
    'has_washer', 'has_dishwasher', 'has_parking', 'has_balcony', 'has_workspace',
]

FEATURES_CAT = [
    'room_type',
    'neighbourhood_cleansed',
    'property_type',
    'host_response_time',
]

DF_CLEANED = ['host_listings_count', 'host_total_listings_count', 'longitude', 'accommodates', 'bedrooms', 'price', 'maximum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'availability_eoy', 'number_of_reviews_ly', 'estimated_occupancy_l365d', 'estimated_revenue_l365d', 'bathrooms', 'has_tv', 'has_kitchen', 'has_ac']

MISSING_FEATURES_NUM = [f for f in FEATURES_NUM if f not in DF_CLEANED] 

print(MISSING_FEATURES_NUM)
print("Nombre de features numériques manquantes : ", len(MISSING_FEATURES_NUM))


# # Conserver uniquement les colonnes présentes dans df
# FEATURES_NUM = [f for f in FEATURES_NUM if f in df.columns]
# FEATURES_CAT = [f for f in FEATURES_CAT if f in df.columns]
# ALL_FEATURES  = FEATURES_NUM + FEATURES_CAT
# TARGET = 'price'
 