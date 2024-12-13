import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import db_manager
import pydeck as pdk
import gdown
import pickle

@st.cache_data
def download_and_load_model():
    file_id = "1fTZrj0iYuG7NEhDHu2WYC_isDjPwhdTW"  # Replace with your Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "random_forest.pkl"

    # Download the model file
    gdown.download(url, output, quiet=False)

    # Load the model
    with open(output, "rb") as f:
        model = pickle.load(f)

    return model

# Load the model
model = download_and_load_model()

saudi_arabia_center = [23.8859, 45.0792]

# Create a Pydeck map
saudi_arabia_map = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v10",
    initial_view_state=pdk.ViewState(
        latitude=saudi_arabia_center[0],
        longitude=saudi_arabia_center[1],
        zoom=5,
        pitch=0,
    ),
)

st.set_page_config(
    page_title="Personalized Tourist Experience Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load data
@st.cache_data
def load_data():
    existing_user_profiles = pd.read_csv('data/user_profiles.csv')
    item_profiles = pd.read_csv('data/item_profiles.csv')
    final_combined_df1 = pd.read_csv('data/final_combined_df1.csv')
    most_pop = pd.read_csv('data/Most_Pop.csv', encoding='latin1')
    return existing_user_profiles, item_profiles, final_combined_df1, most_pop

# Load datasets
existing_user_profiles, item_profiles, final_combined_df1, most_pop = load_data()

# Initialize database for new users
db_manager.create_table()

# Create a mapping of Item ID_tourist to Item Name and Category
item_name_mapping = final_combined_df1.set_index('Item ID_tourist')['Item_name'].to_dict()

# Category mapping
category_mapping = {
    "12 Springs of Prophet Moses (A.S.) - Tabuk Province": "Natural Landscapes",
    "A Trip To The Edge of The World - Riyadh Province": "Natural Landscapes",
    "Abha High City - Asir Province": "Natural Landscapes",
    "Abraj Al Bait Towers - The Clock Towers Complex - Makkah": "Religious Sites",
    "Abu Kheyal Park - Beautiful Hill Top Park in Abha": "Natural Landscapes",
    "Al Bujairi Heritage Park - Riyadh": "Cultural & Heritage Sites",
    "Al Faqir Well - Madinah": "Religious Sites",
    "Al Ghamama Mosque - Madinah": "Religious Sites",
    "Al Ghufran Safwah Hotel Makkah": "Hotels & Accommodations",
    "Al Hair (Ha-ir) Parks and Lakes - Riyadh": "Natural Landscapes",
    "Al Jawatha Historic Mosque - Hofuf - Eastern Province": "Religious Sites",
    "Al Madinah Museum - Hejaz Railway Railway Museum": "Historical Sites",
    "Al Marwa Rayhaan Hotel by Rotana": "Hotels & Accommodations",
    "Al Masjid Al Haram - Makkah": "Religious Sites",
    "Al Masmak Palace Museum": "Historical Sites",
    "Al Qarah Mountain - The Land of Civilisation - Al Ahsa": "Natural Landscapes",
    "Al Rahma - Floating Mosque - Jeddah": "Religious Sites",
    "Al Safwah Royale Orchid Hotel": "Hotels & Accommodations",
    "Al Sahab Park - Asir Province": "Natural Landscapes",
    "Al Shafa Mountain Park - Taif": "Natural Landscapes",
    "Al Shohada Hotel": "Hotels & Accommodations",
    "Al Soudah View Point - Asir Province": "Natural Landscapes",
    "Al Tawba Mosque - Tabuk Province": "Religious Sites",
    "Al Ula - Old Town - Madinah Province": "Cultural & Heritage Sites",
    "Al Wabhah Crater - Makkah Province": "Natural Landscapes",
    "Anjum Hotel Makkah": "Hotels & Accommodations",
    "Anwar Al Madinah Movenpick Hotel": "Hotels & Accommodations",
    "Arch (Rainbow) Rock - Al Ula": "Natural Landscapes",
    "As Safiyyah Museum and Park - Madinah": "Cultural & Heritage Sites",
    "Asfan Fortress - Makkah Province": "Historical Sites",
    "Ash Shafa View Point - Taif": "Natural Landscapes",
    "Baab Makkah Jeddah - Makkah Gate": "Historical Sites",
    "Bir Al Shifa Well - Madinah Province": "Religious Sites",
    "Boulevard - Luxury Shopping Mall - Jeddah": "Shopping & Entertainment",
    "Boulevard Riyadh City - Riyadh": "Shopping & Entertainment",
    "Boulevard World - Riyadh": "Shopping & Entertainment",
    "Catalina Seaplane Wreckage - Tabuk Province": "Historical Sites",
    "Cave at Al Qarah (Jabal) Mountain - Al Ahsa": "Natural Landscapes",
    "Clock Tower Museum - Makkah": "Historical Sites",
    "Cornish Al Majidiyah - Al Qatif - Eastern Province": "Beaches & Waterfronts",
    "Dallah Taibah Hotel": "Hotels & Accommodations",
    "Dammam Corniche - Eastern Province": "Beaches & Waterfronts",
    "Dar Al Madinah Museum - Madinah": "Historical Sites",
    "Desert Ruins of Al Ula - Saudi's Forgotten Past": "Historical Sites",
    "Discover Jabal Ithlib - Archeological Site - Al Ula": "Cultural & Heritage Sites",
    "Discover Jeddah Waterfront on The Red Sea": "Beaches & Waterfronts",
    "Dorrar Aleiman Royal Hotel (Dar Al Eiman Royal Hotel)": "Hotels & Accommodations",
    "Dumat Al Jandal Lake - Al Jawf Province": "Natural Landscapes",
    "Elaf Al Mashaer Hotel Makkah": "Hotels & Accommodations",
    "Elaf Kinda Hotel": "Hotels & Accommodations",
    "Emaar Royal Hotel Al Madinah": "Hotels & Accommodations",
    "Explore New Jeddah Corniche on The Red Sea": "Beaches & Waterfronts",
    "Explore Old Jeddah, Al Balad": "Cultural & Heritage Sites",
    "Fakieh Aquarium - Jeddah": "Modern Attractions",
    "Fanateer Beach - Jubail - Eastern Province": "Beaches & Waterfronts",
    "Four Points by Sheraton Makkah Al Naseem": "Hotels & Accommodations",
    "Harrat Kishb Volcanic Field": "Natural Landscapes",
    "Hegra (Al Hijr) - Mada'in Saleh - Al Ula": "Cultural & Heritage Sites",
    "Hilton Makkah Convention Hotel": "Hotels & Accommodations",
    "Historical At-Turaif World Heritage Site - Riyadh": "Historical Sites",
    "Historical Diriyah Museum - Riyadh": "Historical Sites",
    "Al Madinah Museum - Hejaz Railway Railway Museum": "Historical Sites",
    "Hussainiya Park - Makkah": "Natural Landscapes",
    "Ikmah Heritage Mountain - Jabal Ikmah - Al Ula": "Cultural & Heritage Sites",
    "Jabal Al Nour (Noor) - Mount of Revelation - Makkah": "Religious Sites",
    "Jabal Dakka Park - View Point - Taif": "Natural Landscapes",
    "Jamarat Bridge - Makkah": "Religious Sites",
    "Jannat Al-Mu'alla Cemetry - Makkah": "Religious Sites",
    "Jannatul Baqi Cemetry - Madina": "Religious Sites",
    "Jawatha Park - Hofuf - Eastern Province": "Natural Landscapes",
    "Jeddah Sign": "Modern Attractions",
    "Jeddah Yacht Club and Marina": "Modern Attractions",
    "Khaybar Historical City - Madinah Province": "Historical Sites",
    "Khobar Corniche - Eastern Province": "Beaches & Waterfronts",
    "Khobar Corniche Mosque - Eastern Province": "Religious Sites",
    "King Abdulaziz Center for World Culture - Ithra - Dharan": "Cultural & Heritage Sites",
    "King Abdulaziz Historical Center - National Museum": "Historical Sites",
    "King Abdullah Financial District - KAFD - Riyadh": "Modern Attractions",
    "King Fahad Causeway - Eastern Province": "Modern Attractions",
    "King Fahad's Fountain - Jeddah": "Modern Attractions",
    "King Fahd Glorious Quran Printing Complex - Madinah": "Religious Sites",
    "Kingdom Arena - Riyadh": "Modern Attractions",
    "Kingdom Centre Tower - Mall & Hotel - Riyadh": "Modern Attractions",
    "Lake Park Namar Dam - Riyadh": "Natural Landscapes",
    "M Hotel Makkah Millennium": "Hotels & Accommodations",
    "Makkah Clock Royal Tower, A Fairmont Hotel": "Hotels & Accommodations",
    "Makkah Hotel": "Hotels & Accommodations",
    "Makkah Towers": "Hotels & Accommodations",
    "Marid Castle - Al Jawf Province": "Historical Sites",
    "Masjid Addas - Taif": "Religious Sites",
    "Masjid Al Ijabah - Madinah": "Religious Sites",
    "Masjid Al Jinn - Makkah": "Religious Sites",
    "Masjid Al Qiblatayn - Madinah": "Religious Sites",
    "Masjid Bilal (RA) - Madinah": "Religious Sites",
    "Masjid Imam Bukhari - Madinah": "Religious Sites",
    "Masjid Miqat (Dhul Hulaifah) - Madinah": "Religious Sites",
    "Masjid Quba - Madinah": "Religious Sites",
    "Masjid Sajdah (Abu Dhar Al Ghifari) - Madinah": "Religious Sites",
    "Masjid Shuhada Uhud - Madinah": "Religious Sites",
    "Matbouli House Museum - Jeddah": "Historical Sites",
    "Millennium Makkah Al Naseem Hotel": "Hotels & Accommodations",
    "Mount Arafat - Makkah": "Religious Sites",
    "Mount Uhud - Archers' Hill - Madinah": "Natural Landscapes",
    "Movenpick Hotel & Residence Hajar Tower Makkah": "Hotels & Accommodations",
    "Murabba Historical Palace - Riyadh": "Historical Sites",
    "Murjan Island - Eastern Province": "Beaches & Waterfronts",
    "Nassif House Museum - Jeddah": "Historical Sites",
    "Oud square": "Shopping & Entertainment",
    "Pullman Zamzam Madina": "Hotels & Accommodations",
    "Qanona Valley - Al Bahah Province": "Natural Landscapes",
    "Qantara Mosque (Masjid Madhoon) - Taif": "Religious Sites",
    "Qasab Salt Flats": "Natural Landscapes",
    "Quba Square - Madinah": "Religious Sites",
    "Ras Tanura Beach - Eastern Province": "Beaches & Waterfronts",
    "Red Sand Dunes - Riyadh Province": "Natural Landscapes",
    "Saudi Aramco Exhibit - Al Dhahran": "Historical Sites",
    "Seven Mosques - Madinah": "Religious Sites",
    "Skybridge at Kingdom Centre - Riyadh": "Modern Attractions",
    "Tabuk Castle (Tabuk Fort)": "Historical Sites",
    "Tayma Fort - Tabuk Province": "Historical Sites",
    "The Ain Ancient Village - Al Bahah Province": "Historical Sites",
    "The Ancient City of Madyan - Tabuk Province": "Historical Sites",
    "The Blessed Al Masjid An Nabawi - The Prophet's Mosque (SAWS)": "Religious Sites",
    "The Farasan Islands": "Natural Landscapes",
    "Umluj - Turquoise Waters of the Red Sea": "Natural Landscapes",
    "VIA Riyadh - Riyadh": "Shopping & Entertainment",
    "Yanbu Waterfront - Madinah Province": "Beaches & Waterfronts",
    "Zamzam Well - Makkah": "Religious Sites",
    "ZamZam Pullman Makkah Hotel": "Hotels & Accommodations",
    "Wadi Tayyib Al Ism - Tabuk Province": "Natural Landscapes",
    "Wadi Namer Waterfall - Riyadh": "Natural Landscapes",
    "Wadi Lajab - Jazan Province": "Natural Landscapes",
    "Wadi Laban Dam": "Natural Landscapes",
    "Al Jawatha Historic Mosque - Hofuf - Eastern Province": "Religious Sites",
    "Wadi Disah - Tabuk Province": "Natural Landscapes",
    "Wadi Dharak Lake Park - Al Bahah Province": "Natural Landscapes",
    "Wadi Al Jinn (Baida) - Madinah Province": "Natural Landscapes",
    "Ushaiqer Heritage Village - Riyadh Province": "Cultural & Heritage Sites",
    "Umm Senman Mountain - Hail Province": "Natural Landscapes",
    "Tomb of Lihyan son of Kuza - Al Ula": "Historical Sites",
    "The Spectacular Elephant Rock - Al Ula": "Natural Landscapes",
    "The Seven Mosques (Saba Masajid) - Madinah": "Religious Sites",
    "The Makkah (Mecca) Museum": "Cultural & Heritage Sites",
    "The Ghars Well - Madinah - بئر غرس": "Natural Landscapes",
    "The Garden of Salman Al Farsi (RA) - Madinah": "Cultural & Heritage Sites",
    "The Ethiq Well and Garden - Madinah": "Cultural & Heritage Sites",
    "The Art Street - Abha": "Modern Attractions",
    "Tarout Island - Eastern Province": "Historical Sites",
    "Taif Cable Cars (Telefric Al Hada)": "Modern Attractions",
    "Taiba Hotel Madinah": "Hotels & Accommodations",
    "Tabuk Ottoman Castle": "Historical Sites",
    "Jabbal Dakka Park": "Natural Landscapes",
    "Swissotel Makkah": "Hotels & Accommodations",
    "Swissotel Al Maqam Makkah": "Hotels & Accommodations",
    "Sparhawk Arch -Rock Formation - Al Ula": "Historical Sites",
    "Sofitel Shahd al Madinah": "Hotels & Accommodations",
    "Sky Bridge Kingdom Tower": "Modern Attractions",
    "Saqifah Bani Saidah - Madinah": "Cultural & Heritage Sites",
    "Riyadh Zoo Day Trip - Riyadh Province": "Modern Attractions",
    "Rijal Almaa - Beautiful Heritage Village": "Cultural & Heritage Sites",
    "Quba Walkway Park- Madinah": "Modern Attractions",
    "Quba Square - Madinah - ساحة قباء": "Modern Attractions",
    "Oud square - عُود سكوير": "Modern Attractions",
    "Nassif House Museum -Jeddah": "Cultural & Heritage Sites",
    "Lake Park Namar Dam -Riyadh": "Natural Landscapes",
    "King Abdullah Financial District  - KAFD - Riyadh": "Modern Attractions",
    "Jabal Dakka Park  - Taif - جبل دكا": "Natural Landscapes",
    "Bir Al Shifa Well - Madinah Province - بئر الشفاء": "Cultural & Heritage Sites",
    "Al Madinah Museum - Hejaz Railway Railway Museum - متحف السكة الحديد": "Cultural & Heritage Sites"
}

# Fetch new users from the SQLite database
def fetch_new_users():
    db_user_profiles = pd.DataFrame(
        db_manager.fetch_all_users(),
        columns=['username', 'user_id', 'preferred_province', 'category_of_interest', 'activity_level']
    )
    return db_user_profiles

# Set default tab to Home
if 'tab' not in st.session_state:
    st.session_state['tab'] = "Home"

# App Title
st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        font-family: 'Arial', sans-serif;
        color: #6ca068;
        font-weight: bold;
        text-align: center;
    }
    .description {
        font-size: 18px;
        font-family: 'Verdana', sans-serif;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <div class="title">Personalized Tourist Experience Recommender</div>
    <div class="description">Explore and get recommendations for the best attractions in Saudi Arabia!</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for Login functionality
with st.sidebar:
    if 'user_id' not in st.session_state:
        st.write("### Log In or Create an Account")
        user_id = st.text_input("Enter User ID:")

        if st.button("Submit"):
            user_id = user_id.strip().lower()

            # Check if the user exists in the CSV
            if user_id in existing_user_profiles['User ID'].str.lower().values:
                st.session_state['user_id'] = user_id
                st.session_state['tab'] = "Home"
                st.success("Welcome back!")
            else:
                # Check if the user exists in the database
                user_profile = db_manager.fetch_user_by_id(user_id)
                if user_profile:
                    username = user_profile[0]  # Fetch username from the database record
                    st.session_state['user_id'] = user_id
                    st.session_state['tab'] = "Home"
                    st.success(f"Welcome {username}!")
                else:
                    # Flag for new user account creation
                    st.session_state['new_user'] = True

        # Only show account creation for new users
        if 'new_user' in st.session_state and st.session_state['new_user']:
            st.write("### New User Account Creation")
            new_username = st.text_input("Enter Username:")
            new_preferred_province = st.selectbox("Preferred Province:", ['All'] + list(item_profiles['Province'].dropna().unique()))
            new_category_of_interest = st.selectbox("Category of Interest:", ['All'] + list(set(category_mapping.values())))
            new_activity_level = st.slider("Activity Level (1 to 5):", min_value=1, max_value=5, value=3)

            if st.button("Create Account"):
                if new_username:
                    db_manager.insert_user(new_username, user_id, new_preferred_province, new_category_of_interest, new_activity_level)
                    st.success("Account created successfully!")
                    st.session_state['user_id'] = user_id
                    st.session_state['tab'] = "Home"
                    del st.session_state['new_user']  # Remove new user flag
                else:
                    st.error("Username cannot be empty.")
    elif 'user_id' in st.session_state:
        st.write(f"Logged in as: {st.session_state['user_id']}")
        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.stop()



selected = option_menu(
    menu_title=None,
    options=["Home", "Recommendation Engine", "Popular Attractions"],
    icons=["house", "compass", "map"],
    menu_icon="cast",
    default_index=["Home", "Recommendation Engine", "Popular Attractions"].index(st.session_state['tab']),
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f9f9f9"},
        "icon": {"color": "black", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#9d69a7"},
    }
)

if selected == "Home":
    st.markdown("### *Discover Saudi Arabia's Best Attractions!*")
    st.image("images/banar.jpg", use_container_width=True)
    st.markdown(" **Saudi Arabia is a land of beauty and culture. The Kingdom has become one of the most important"
             " tourist destinations in the world due to its distinguished geographical location and rich cultural heritage.**")
    st.pydeck_chart(saudi_arabia_map)

elif selected == "Recommendation Engine":
    st.markdown("### *Get tailored recommendations based on your preferences!*")
    if 'user_id' not in st.session_state:
        st.error("Please login to use this engine.")
    else:
        user_id = st.session_state['user_id']

        # Fetch user profile
        if user_id in existing_user_profiles['User ID'].str.lower().values:
            user_profile = existing_user_profiles[existing_user_profiles['User ID'].str.lower() == user_id].iloc[0]
            activity_level = user_profile['activity_level']
            preferred_province = user_profile['Province'] if 'Province' in user_profile else 'All'
            category_of_interest = user_profile[
                'Category of Interest'] if 'Category of Interest' in user_profile else 'All'
        else:
            user_profile = db_manager.fetch_user_by_id(user_id)
            if user_profile:
                activity_level = user_profile[4]
                preferred_province = user_profile[2] or 'All'
                category_of_interest = user_profile[3] or 'All'
            else:
                st.error("No valid user profile found. Please log in or create an account.")
                st.stop()

        # Filter item profiles based on user preferences
        filtered_item_profiles = item_profiles.copy()

        # Add filtering options
        province_filter = st.selectbox(
            "Filter by Province:",
            options=['All'] + list(filtered_item_profiles['Province'].dropna().unique()),
            index=['All'] + list(filtered_item_profiles['Province'].dropna().unique()).index(preferred_province)
            if preferred_province in list(filtered_item_profiles['Province'].dropna().unique()) else 0
        )

        category_filter = st.selectbox(
            "Filter by Category:",
            options=['All'] + sorted(set(category_mapping.values())),
            index=['All'] + sorted(set(category_mapping.values())).index(category_of_interest)
            if category_of_interest in set(category_mapping.values()) else 0
        )

        if province_filter != 'All':
            filtered_item_profiles = filtered_item_profiles[filtered_item_profiles['Province'] == province_filter]

        if category_filter != 'All':
            filtered_item_profiles['Item_name'] = filtered_item_profiles['Item ID_tourist'].map(item_name_mapping)
            filtered_item_profiles['Item Category'] = filtered_item_profiles['Item_name'].map(category_mapping)
            filtered_item_profiles = filtered_item_profiles[filtered_item_profiles['Item Category'] == category_filter]

        if filtered_item_profiles.empty:
            st.warning("No recommendations found for the selected filters.")
        else:
            # Generate feature inputs for the Random Forest model
            svd_scores = np.random.rand(len(filtered_item_profiles))  # Example SVD scores
            user_features = np.array([activity_level] * 2).reshape(1, -1)
            item_features = filtered_item_profiles[['City_Sentiment_Score', 'Avg_City_Sentiment_Score']].values
            kbf_scores = cosine_similarity(user_features, item_features).flatten()

            features = pd.DataFrame({
                'SVD Score': svd_scores,
                'KBF Score': kbf_scores
            })

            # Predict interaction probabilities
            predictions = model.predict_proba(features)[:, 1]  # Probability of interaction
            filtered_item_profiles['Predicted Score'] = predictions

            # Sort recommendations
            recommendations = filtered_item_profiles.sort_values(by='Predicted Score', ascending=False).head(5)

            # Display recommendations
            if recommendations.empty:
                st.warning("No recommendations found for the selected filters.")
            else:
                recommendations['Item_name'] = recommendations['Item ID_tourist'].map(item_name_mapping)
                recommendations['Item Category'] = recommendations['Item_name'].map(category_mapping)
                recommendations['Rank'] = range(1, len(recommendations) + 1)
                recommendations = recommendations[['Rank', 'Item_name', 'City', 'Item Category']]
                st.markdown(recommendations.to_markdown(index=False))
elif selected == "Popular Attractions":
    st.write("### Explore the Popular Attractions")
    attraction_name = st.selectbox("Select an Attraction:", most_pop['Item_name'])
    if attraction_name:
        attraction_details = most_pop[most_pop['Item_name'] == attraction_name].iloc[0]
        st.image(attraction_details['Image_path'], caption=attraction_name, width=600)
        st.write(attraction_details['Description'])
        st.write("For more information, [Visit Saudi](https://www.visitsaudi.com)")