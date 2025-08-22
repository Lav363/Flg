from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
try:
    model = tf.keras.models.load_model("flag_model.h5")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# Load class labels
try:
    class_names = sorted(os.listdir("country-flags-in-the-wild/train"))
    if not class_names:
        raise ValueError("Class names folder is empty.")
    logger.info(f"✅ Class names loaded: {class_names}")
except Exception as e:
    logger.error(f"❌ Error loading class names: {e}")
    class_names = []

# Sample mapping from numeric class labels to country names
# You should update this dictionary with the correct mappings
class_label_to_country = {

    '1': 'Afghanistan',
    '2': 'Albania',
    '3': 'Algeria',
    '4': 'American Samoa',
    '5': 'Andorra',
    '6': 'Angola',
    '7': 'Anguilla',
    '8': 'Antigua and Barbuda',
    '9': 'Argentina',
    '10': 'Armenia',
    '11': 'Aruba',
    '12': 'Australia',
    '13': 'Austria',
    '14': 'Azerbaijan',
    '15': 'Bahamas',
    '16': 'Bahrain',
    '17': 'Bangladesh',
    '18': 'Barbados',
    '19': 'Belarus',
    '20': 'Germany',
    '21': 'Belize',
    '22': 'Benin',
    '23': 'Bermuda',
    '24': 'Bhutan',
    '25': 'Bolivia',
    '26': 'Bosnia and Herzegovina',
    '27': 'Botswana',
    '28': 'Brazil',
    '29': 'British Virgin Islands',
    '30': 'Brunei Darussalam ',
    '31': 'Bulgaria',
    '32': 'Burkina Faso',
    '33': 'Burundi',
    '34': 'Cambodia',
    '35': 'Cameroon',
    '36': 'Canada',
    '37': 'Cabo Verde',
    '38': 'Cayman Islands',
    '39': 'Central African Republic',
    '40': 'Romania',
    '41': 'Chile',
    '42': 'China',
    '43': 'Solomon Islands',
    '44': 'Colombia',
    '45': 'Comoros',
    '46': 'Tuvalu',
    '47': 'Thailand',
    '48': 'Cte dIvoire',
    '49': 'Croatia',
    '50': 'Cuba',
    '51': 'Cyprus',
    '52': 'Turkish Republic of Northern Cyprus',
    '53': 'Czech Republic',
    '54': 'Congo, Democratic Republic of the',
    '55': 'Denmark',
    '56': 'Djibouti',
    '57': 'Dominica',
    '58': 'Dominican Republic',
    '59': 'Ecuador',
    '60': 'Egypt',
    '61': 'El Salvador',
    '62': 'Equatorial Guinea',
    '63': 'Eritrea',
    '64': 'Botswana',
    '65': 'Ethiopia',
    '66': 'Falkland Islands',
    '67': 'Faroe Islands',
    '68': 'Fiji',
    '69': 'Finland',
    '70': 'France',
    '71': 'French Polynesia',
    '72': 'Gabon',
    '73': 'Gambia',
    '74': 'Georgia',
    '75': 'Germany',
    '76': 'Ghana',
    '77': 'Gibraltar',
    '78': 'Greece',
    '79': 'Greenland',
    '80': 'Grenada',
    '81': 'Guam',
    '82': 'Guatemala',
    '83': 'Guinea',
    '84': 'Guinea-Bissau',
    '85': 'Guyana',
    '86': 'Haiti',
    '87': 'Honduras',
    '88': 'Hong Kong',
    '89': 'Hungary',
    '90': 'Iceland',
    '91': 'India',
    '92': 'Indonesia',
    '93': 'Islamic Republic of Iran',
    '94': 'Iraq',
    '95': 'Ireland',
    '96': 'Israel',
    '97': 'Italy',
    '98': 'Jamaica',
    '99': 'Japan',
    '100': 'Jordan',
    '101': 'Kazakhstan',
    '102': 'Kenya',
    '103': 'Kiribati',
    '104': 'Kuwait',
    '105': 'Kyrgyzstan',
    '106': 'Laos',
    '107': 'Austria',
    '108': 'Lebanon',
    '109': 'Lesotho',
    '110': 'Liberia',
    '111': 'Libya',
    '112': 'Liechtenstein',
    '113': 'Lithuania',
    '114': 'Luxembourg',
    '115': 'Macau',
    '116': 'North Macedonia',
    '117': 'Madagascar',
    '118': 'Malawi',
    '119': 'Malaysia',
    '120': 'Maldives',
    '121': 'Mali',
    '122': 'Malta',
    '123': 'Marshall Islands',
    '124': 'Martinique',
    '125': 'Mauritania',
    '126': 'Mauritius',
    '127': 'Mexico',
    '128': 'Micronesia',
    '129': 'Moldova',
    '130': 'Indonesia',
    '131': 'Mongolia',
    '132': 'Montserrat',
    '133': 'Morocco',
    '134': 'Mozambique',
    '135': 'Myanmar',
    '136': 'Namibia',
    '137': 'Nauru',
    '138': 'Nepal',
    '139': 'Netherlands',
    '140': 'Netherlands Antilles',
    '141': 'New Zealand',
    '142': 'Nicaragua',
    '143': 'Niger',
    '144': 'Nigeria',
    '145': 'Niue',
    '146': 'Norfolk Island',
    '147': 'North Korea',
    '148': 'Norway',
    '149': 'Oman',
    '150': 'Pakistan',
    '151': 'Palau',
    '152': 'Panama',
    '153': 'Papua New Guinea',
    '154': 'Paraguay',
    '155': 'Peru',
    '156': 'Philippines',
    '157': 'Pitcairn Islands',
    '158': 'Poland',
    '159': 'Portugal',
    '160': 'Puerto Rico',
    '161': 'Qatar',
    '162': 'Republic of the Congo',
    '163': 'Romania',
    '164': 'Russia',
    '165': 'Rwanda',
    '166': 'Saint Kitts and Nevis',
    '167': 'Saint Lucia',
    '168': 'Saint Pierre and Miquelon',
    '169': 'Saint Vincent and the Grenadines',
    '170': 'Samoa',
    '171': 'San Marino',
    '172': 'São Tomé and Príncipe',
    '173': 'Saudi Arabia',
    '174': 'Senegal',
    '175': 'Yugoslavia',
    '176': 'Seychelles',
    '177': 'Sierra Leone',
    '178': 'Singapore',
    '179': 'Slovakia',
    '180': 'Slovenia',
    '181': 'Solomon Islands',
    '182': 'Somalia',
    '183': 'South Africa',
    '184': 'South Georgia and the South Sandwich Islands',
    '185': 'South Korea',
    '186': 'Soviet Union',
    '187': 'Spain',
    '188': 'Sri Lanka',
    '189': 'Sudan',
    '190': 'Suriname',
    '191': 'Eswatini (Swaziland)',
    '192': 'Sweden',
    '193': 'Switzerland',
    '194': 'Syria',
    '195': 'China',
    '196': 'Tajikistan',
    '197': 'Tanzania',
    '198': 'Thailand',
    '199': 'Tibet',
    '200': 'Timor-Leste',
    '201': 'Togo',
    '202': 'Tonga',
    '203': 'Trinidad and Tobago',
    '204': 'Tunisia',
    '205': 'Turkey',
    '206': 'Turkmenistan',
    '207': 'Turks and Caicos Islands',
    '208': 'Tuvalu',
    '209': 'United Arab Emirates ',
    '210': 'Uganda',
    '211': 'Ukraine',
    '212': 'United Kingdom',
    '213': 'United States of America',
    '214': 'Uruguay',
    '215': 'United States Virgin Islands',
    '216': 'Uzbekistan',
    '217': 'Vanuatu',
    '218': 'Vatican City',
    '219': 'Venezuela',
    '220': 'Vietnam',
    '221': 'Wallis and Futuna',
    '222': 'Yemen',
    '223': 'Zambia',
    '224': 'Zimbabwe'
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model is not loaded.", 500
    if not class_names:
        return "Class names not loaded.", 500
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    filename = file.filename
    img_path = os.path.join("static", filename)

    try:
        os.makedirs("static", exist_ok=True)
        file.save(img_path)

        # Preprocess image
        img = load_img(img_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_label = class_names[class_index]
        country_name = class_label_to_country.get(predicted_label, predicted_label).replace("_", " ").title()

        return render_template("index.html", prediction=country_name, image_file=filename)
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return f"Prediction failed: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
