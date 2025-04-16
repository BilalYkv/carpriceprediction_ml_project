import streamlit as st
import pickle
from PIL import Image
import numpy as np


def main():
    # Load the trained model and scaler
    model = pickle.load(open('Random_Forest_model.sav', 'rb'))
    scaler = pickle.load(open('scaler_model.sav', 'rb'))

    st.title("🚗 Car Price Prediction")
    image = Image.open('carprice.jpg')
    st.image(image, caption='Car Price Prediction', width=300)

    # User Inputs
    brand_models = ['Maruti','Hyundai','Mahindra','Tata','Toyota','Honda','Ford','Chevrolet','Renault',
                         'Volkswagen','BMW','Skoda','Nissan','Jaguar','Volvo','Datsun','Mercedes-Benz','Fiat'
                         'Audi','Lexus','Jeep','Mitsubishi','Force''Land','Isuzu','Kia','Ambassador','Daewoo',
                         'MG','Ashok','Opel''Peugeot']
    brand = st.selectbox("Select the Car Brand", ["Select"] + list(brand_models))

    year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)

    fuel_type_display = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
    fuel_type_encoded = [0, 1, 2, 3, 4]
    fuel_type = st.selectbox("Select the fuel type", ["Select"] + fuel_type_display)

    if fuel_type != "Select":
        fuel_type = fuel_type_encoded[fuel_type_display.index(fuel_type)]
    else:
        fuel_type = None

    seller_type_display = ["Dealer", "Individual", "Trustmark Dealer"]
    seller_type_encoded = [0, 1, 2]
    seller_type = st.selectbox("Select the seller type", ["Select"] + seller_type_display)

    if seller_type != "Select":
        seller_type = seller_type_encoded[seller_type_display.index(seller_type)]
    else:
        seller_type = None

    transmission_display = ["Manual", "Automatic"]
    transmission_encoded = [1, 0]
    transmission = st.selectbox("Select the transmission type", ["Select"] + transmission_display)

    if transmission != "Select":
        transmission = transmission_encoded[transmission_display.index(transmission)]
    else:
        transmission = None

    owner_type_display = ["First", "Second", "Third", "Fourth & Above"]
    owner_type_encoded = [0, 1, 2, 3]
    owner_type = st.selectbox("Select the owner type", ["Select"] + owner_type_display)

    if owner_type != "Select":
        owner_type = owner_type_encoded[owner_type_display.index(owner_type)]
    else:
        owner_type = None

    mileage = st.number_input("Mileage (km/l or km/kg)", min_value=5.0, step=0.1)
    engine = st.number_input("Engine Size (CC)", min_value=800, step=100)
    power = st.number_input("Max Power (bhp)", min_value=30.0, step=0.1)
    seats = st.selectbox("Number of Seats", ["Select", 2, 4, 5, 6, 7, 8, 10])

    if seats == "Select":
        seats = None

    if None not in [brand, fuel_type, seller_type, transmission, owner_type, seats]:
        # Feature Scaling
        features = np.array([
            year, kilometers_driven, fuel_type, seller_type, transmission,
            owner_type, mileage, engine, power, seats
        ]).reshape(1, -1)

        features = scaler.transform(features)

        if st.button("Predict Price"):
            prediction = model.predict(features)
            st.success(f"💰 The estimated price of the {brand} is: ₹{prediction[0]:,.2f}")


if __name__ == "__main__":
    main()