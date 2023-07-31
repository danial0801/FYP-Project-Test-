import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu 


# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\Final Year Project\Saved Models\Models Split(8.1.1)\MobileNet\MobileNet Model_best.h5")

# Define the class labels
class_labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 
                'bacterial_panicle_blight', 'blast', 'brown_spot', 
                'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']  # Add your class labels here

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Paddy Disease Classification',
                          
                          ['Paddy Disease', 
                           'Classifier'],
                          icons=[ 'person','activity'],
                          default_index=0)

# Classifier
if (selected == 'Classifier'):
    # Function to preprocess the image
    def preprocess_image(image):
        img = image.resize((128, 128))  # Resize the image to match the input size of the model
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = tf.keras.applications.mobilenet.preprocess_input(img_array)
        return preprocessed_img

    # Function to make predictions
    def make_prediction(image):
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        return predicted_class, confidence



    # Streamlit app
    st.title("Paddy Disease Classification Prototype")
    st.write("Upload an image for classification")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make predictions on uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_class, confidence = make_prediction(image)
        st.write("Predicted Class:", predicted_class)
        st.write("Confidence:", confidence)
        
# Classifier
if (selected == 'Paddy Disease'):
    title_text = "Paddy Disease" 
    centered_title_html = f"<h1 style='text-align: center;'>{title_text}</h1>"
    st.write(centered_title_html, unsafe_allow_html=True)
    st.write("") 
    st.write(f"<div style='text-align: center;'>This project focuses on classifing 9 different types of paddy diseases with one\
             healthy paddy class. Agriculture plays a huge role in sustaining the life of\
            every  human  being  by  providing  sustenance and food.Over 90% of the world’s rice is produced and\
            consumed in South-East Asia. Paddy plants are susceptible to diseases and infections by viruses, \
            fungi and bacteria. Paddy diseases could causes the quality of these paddies to decrease along with their value of\
            products. In Malaysia, the consumption per capita for Malaysian citizen was 87.9 Kg per person in \
            the year 2016 which is much higher than the global average of 54.6 Kg/person that year\
            (S. C. Omar et al., 2019). The production of rice in Malaysia was over 1,512,709 metric \
            tonnes in the year 2020. These facts support the notion that rice is a major part of the \
            Malaysian diet and is considered a staple food for the citizens (Zakaria & Nik Abdul \
            Ghani, 2022).The destruction of paddy leaves may be caused by the \
            atmospheric situation, nutrient issues, social issues, and substances. Pathogens such as \
            bacteria, viruses, and fungi contribute to the infection of diseases in paddy leaves (Payal \
            et al., 2020). These diseases infect the paddy leaves causing the quality of these paddies \
            to decrease along with their value of products which will also generate the problem of \
            low rice supply.</div>", unsafe_allow_html=True)
    st.write("")       
    st.write("") 
    st.write("") 
    
    
    #show image in circle for each disease
    def image_circle(image_url, size=150):
        st.markdown(
        f"""
        <style>
        .circle-image {{
            width: {size}px;
            height: {size}px;
            border-radius: {size/2}px;
            overflow: hidden;
        }}
        .circle-image img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: {size/2}px;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
        st.markdown(f"<div class='circle-image'><img src='{image_url}'></div>", unsafe_allow_html=True)
    
    
    col1, col2= st.columns(2)
    
    with col1:
        options = ['Choose a disease','bacterial_leaf_blight', 'bacterial_leaf_streak', 
         'bacterial_panicle_blight', 'blast', 'brown_spot', 
         'dead_heart', 'downy_mildew', 'hispa', 'tungro']
        
        selected_option = st.radio('Choose an option', options)

    with col2: 
        if selected_option == 'Choose a disease': 
            image_url = r"C:\Users\Asus\Downloads\de-an-sun-ih7n_6wsEIg-unsplash(1).jpg"
            st.image(Image.open(image_url))
        
        if selected_option == 'bacterial_leaf_blight': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Bacterial Leaf Streak\100084.jpg"
            st.image(Image.open(image_url))
            st.write('**Bacterial leaf Blight**')
            st.write(" Bacterial leaf blight is caused by infections by pathogens named Xanthomonas oryzae PV. \
                     . It affects the leaf or panicle in that it causes its seedlings to wilt and the leaf to \
                    turn yellow and dry. According to the International Rice Research Institute (IRRI), the disease is \
                    most prone to spread to areas with weeds and contaminated plant residue. The \
                    organization states that it can appear in both tropical and temperate settings, especially \
                    in lowland areas that receive irrigation and rainfall. The disease prefers, generally, \
                    temperatures between 25 and 34 °C, with a relative humidity of at least 70%")
                            
        if selected_option == 'bacterial_leaf_streak': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Bacterial Leaf Streak\100150.jpg"
            st.image(Image.open(image_url))
            st.write('**Bacterial leaf Streak**')
            st.write(" Bacterial leaf streak disease is mainly attributed to infections by Xanthomonas \
                     oryzae pv. oryzicola. The infected paddy leaf would manifest symptoms of browning \
                    and dryness which will cause the weight of the grain to decrease (G. Premi et al., 2019). \
                    The official website of the IRRI for Rice Knowledge indicates that leaf streaks transpire \
                    in areas that are high in temperature and humidity. The approach by which the infection \
                    occurs is through infected carrier seed and stubble for the next planting season. This \
                    infection could also transpire if the leaves, water, and debris of the field if there are \
                    presence of said bacteria. The degree of yield loss consequent of this disease depends \
                    on the climate with the loss being 8-17% during rainy seasons and 1-3% during dry \
                    seasons. The microorganism would infect the plant if it were present within the water \
                    canal or if it is on the leaf of the plant itself")
        
        if selected_option == 'bacterial_panicle_blight': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Bacterial Panicle Blight\100071.jpg"
            st.image(Image.open(image_url))
            st.write('**Bacterial Panicle Blight**')
            st.write(" Bacterial Panicle blight is caused by a type of bacterium called Burkholderia glumae. \
                     . It affects the leaf or panicle in that it causes its seedlings to wilt and the leaf to \
                    turn yellow and dry. According to the International Rice Research Institute (IRRI), the disease is \
                    most prone to spread to areas with weeds and contaminated plant residue. The \
                    organization states that it can appear in both tropical and temperate settings, especially \
                    in lowland areas that receive irrigation and rainfall. The disease prefers, generally, \
                    temperatures between 25 and 34 °C, with a relative humidity of at least 70%")       

        if selected_option == 'blast': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Blast\100004.jpg"
            st.image(Image.open(image_url))
            st.write('**Bacterial Panicle Blight**')
            st.write(" Rice blast infections are attributable to infection from the fungus Magnaporthe \
                    oryzae or filamentous ascomycete (Asibi et al., 2019; International Rice Research \
                    Institute, n.d.). The IRRI stated that the disease strives in regions with low soil moisture, \
                    frequent and protracted rainfall, and chilly daytime temperatures. Large day-night \
                    temperature fluctuations that result in dew forming on leaves and generally lower \
                    temperatures in upland rice enhance and promote the growth of the blast disease. The \
                    blast disease could infect multiple parts of the paddy plant which includes the leaf, \
                    collar, node, and neck parts of the plant. The typical symptom of leaf blasts is \
                    characterized by elliptical spots with a spindle-like shape and a tapered end. The spots' \
                    margins are often dark or reddish-brown, with a greyish or whitish centre (International \
                    Rice Research Institute, n.d.). The fungus is most common on the leaves, where it causes \
                    leaf blasting during vegetative development stages, or on the necks and panicles during \
                    the reproductive stage (Shahriar et al., 2020). The authors also mentioned that the yield \
                    loss due to the blast disease epidemic could get as high as 60% to 65% of product yield.")

        if selected_option == 'brown_spot': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Brown Spot\100022.jpg"
            st.image(Image.open(image_url))
            st.write('**Brown Spot**')
            st.write(" Brown spots in paddy plantations are one of the most common types of disease \
                     there are for paddy harvest and are hailed as one of the most damaging diseases for \
                    paddy plants. The multiple large blotches on the leaves that can kill the entire leaf are \
                    the most noticeable harm (International Rice Research Institute, n.d.). The organization \
                    also states that this disease usually infects the coleoptile, leaves, leaf sheath, panicle \
                    branches, glumes, and spikelets. This ailment occurs in high-humidity areas reaching \
                    86% to 100% relative humidity along with nutrient-deficient or toxic soil (Taujuddin et \
                    al., 2021). If not treated the brown spot disease could be very detrimental to the quality \
                    and quantity of the supposed paddy yield as it causes both quantity and quality loss. The \
                    degree of loss depends on the degree of infection in the plants or seedlings. For heavily infected ones, the loss can reach up to 45% and in more moderate instances, the loss \
                    still gets as high as 12%.")
                    

        if selected_option == 'dead_heart': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Dead Heart\100036.jpg"
            st.image(Image.open(image_url))
            st.write('**Dead Heart**')
            st.write(" Dead hearts are a significant product of an attack by a type of pest labelled as \
                    stem borers. According to the IRRI, these pests inflict damage to the plant’s overall \
                    health by boring inside the stem portion of the paddy plant during its vegetative stage. \
                    There are a variety of stem borer species which includes yellow stemborer, white \
                    stemborer, striped stemborer, gold-fringed stemborer, dark-headed striped stemborer, \
                    and pink stemborer. These pests would bore on the inner portion of the leaf sheath \
                    causing tunnelling in the stem of the paddy plant. Consequently, the central leaf whorl \
                    would turn brown, refuse to open, and dry up (Dey, 2020). This condition is therefore \
                    described as a “Dead Heart”. In addition, according to the author, this symptom is also \
                    followed by a condition called “White Ear Head” which means a paddy plant with a \
                    large panicle but empty grains.")
                    
        if selected_option == 'downy_mildew': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Downey Mildew\100017.jpg"
            st.image(Image.open(image_url))
            st.write('**Downy Mildew**')
            st.write(" Downy Mildew is a kind of plant disease that affects crops yield quantity and \
                    quality. It can be caused by infections by pathogens and have a variety of species which \
                    includes Plasmopara viticola, Pseudoperonospora cubensis, Pseudoperonospora \
                    humuli, and Peronospora belbahrii (Salcedo et al., 2021). In accordance with the IRRI, \
                    early symptoms of the illness include chlorotic whitish spots or patches on new leaves \
                    of infected plants, which can sometimes cause the size of the leaf blades to decrease. \
                    On the underside of the leaf blades, fluffy white to grey fungal growth can occasionally \
                    be visible. Young shoots and seeds that have been infected may also have a white fungal \
                    growth coating. The organization also mentioned that in the case where the infection is \
                    more severe, the leaf may appear to be twisted or distorted. The panicle of the paddy \
                    would also be smaller than usual with it being green for longer than normal .")

        if selected_option == 'hispa': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Hispa\100187.jpg"
            st.image(Image.open(image_url))
            st.write('**Hispa**')
            st.write(" Instead of pathogens, bacterium, or fungi, the Hispa disease is caused by \
                    scraping on the upper parts of the paddy leaves by an insect pest known as Dicladispa \
                    armigera or more commonly known as Rice Hispa (Sharma et al., 2017). This will cause \
                    the epidermis of the paddy leaves to thin and scrape, creating producing a whitish streak. \
                    If the severity of damage is high, then it will decrease and lower the paddy plants’ \
                    vitality. Proportional to the IRRI official website, the rice hispa is widespread in wetland \
                    habitats that are both rainfed and irrigated, and it is more prevalent during the rainy \
                    season. As alternative hosts, grassy weeds in and around rice fields harbour and promote \
                    the pest's growth. A field that has been too heavily fertilized also increases the \
                    deterioration of the paddy that has been infected by the rice hispa pests. The extent of \
                    damage the rice hispa could do to paddy corps reaches up to 20% of yield loss on \
                    average but could potentially reach up to 52% in deep water rice.")
                    
        if selected_option == 'tungro': 
            image_url = r"C:\Final Year Project\Dataset\Excluded Dataset (Predictions)\Tungro\100060.jpg"
            st.image(Image.open(image_url))
            st.write('**Tungro**')
            st.write(" Tungro is a type of disease that affects infected paddy plants in that they would \
                    experience reduced tillering and stunted growth (Msg. Premi, 2019). This disease is \
                    attributed to the combined infections of two different viruses, the Rice Tungro Spherical \
                    Virus (RTSV) and Rice Tungro Bacilliform Virus (RTBV). According to the IRRI \
                    organization’s official website, by feeding on plants with tungro disease, leafhoppers \
                    can spread the virus from one plant to another. After the leafhoppers are fed on the \
                    tungro-infected paddy plant, they can transmit the disease to other paddy plants within \
                    5-7 days before the viral load decreases. The yellow or orange-yellow discolouration\
                    that extends from the leaf tip to the blade or lower portion of the leaf is the infection's \
                    typical symptom. Stunting, rust-coloured patches, fewer tillers, and a striped appearance \
                    are other signs and symptoms of a tungro-infected plant. In South and Southeast Asia, \
                    tungro is one of the most harmful and devastating diseases of rice (International Rice \
                    Research Institute, n.d.). Tungro-sensitive cultivars that are infected at an early growth \
                    stage could suffer up to a 100% yield loss in extreme circumstances .")                    
