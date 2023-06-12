import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import pandas as pd
import time

# download and store the images


def download_image(url, directory):

    response = requests.get(url)
    if response.status_code == 200:
        with open(directory, 'wb') as file:
            file.write(response.content)
            print(f"Image downloaded: {directory}")


def scrape_unsplash_images(keyword, num_images, directory, links_csv_path):

    # if a few images have already been downloaded, store their links in an array so that we dont download them again.
    csv_path = links_csv_path
    if os.path.isfile(csv_path):
        scraped_urls = np.array(pd.read_csv(csv_path))
    else:
        scraped_urls = []

    scraped_urls = list(map(lambda x: x[0], scraped_urls))
    scraped_urls = scraped_urls[1:]

    url = f"https://unsplash.com/s/photos/{keyword}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    image_tags = soup.find_all('img')

    i = 0
    for tag in image_tags:

        if 'class' in tag.attrs and 'srcset' in tag.attrs and len(tag.attrs['class']) == 2 and 'tB6UZ' in tag.attrs['class']:
            links = tag.attrs['srcset'].split(', ')

            for link in links:
                link = link.split(' ')
                if link[1] == '900w':
                    image_url = link[0]
                    image_name = f"{directory}/{keyword}_{i+1}.jpg"

                    if image_url not in scraped_urls:
                        if 'plus' in image_url:
                            return

                        download_image(image_url, image_name)
                        scraped_urls.append(image_url)
                        i += 1
                    break

        if i >= num_images:
            break

    df = pd.DataFrame(scraped_urls)
    df.to_csv(csv_path, index=False, header=True)


# list of all the words to search.
list_of_list_of_queries = [
    ['truck', 'Pickup truck', 'Ford F-150', 'Flatbed truck', 'Refrigerator truck', 'Dump truck', 'Garbage truck', 'Tank truck', 'Freightliner Trucks', 'Box truck', 'Chevrolet Silverado', 'Tractor unit', 'RAM 1500',
        'Light truck', 'Car carrier trailer', 'Toyota Tacoma', 'Semi-trailer truck', 'Refrigerated container', 'RAM 2500', 'Commercial vehicle', 'Ford F-150 Electric', 'Chassis cab', 'Trailer', 'Haul truck', 'UD Trucks'],
    ['ships', 'Container ship', 'Bulk carrier', 'Frigate', 'Cruiser', 'Schooner', 'Destroyer', 'Tanker', 'Cargo ship', 'Galleon', 'Passenger ship', 'Corvette', 'Sloop',
        'Battleship', 'Fishing vessel', 'Tugboat', 'Barge', 'Brig', 'Roll-on/roll-off', 'Submarine', 'Barque', 'Battlecruiser', 'Oil tanker', 'Galley', 'Cruise ship'],
    ['frogs', 'True toad', 'American bullfrog', 'Glass frogs', 'South American horned frogs', 'Red-eyed tree frog', 'Desert rain frog', 'True frog', 'Tree frogs', 'Poison dart frog', 'Australian green tree frog', 'Goliath frog', 'African dwarf frog', 'Cane toad', 'Breviceps adspersus', 'Common eastern froglet', 'Banded bullfrog', 'Common coquí', 'Giant ditch frog', 'Kaloula', 'Breviceps fuscus', 'Telmatobius culeus', 'Rain frogs', 'Lepidobatrachus laevis', 'Mission golden-eyed tree frog', 'Neobatrachia',
        'African clawed frog', 'Suriname Toad', 'African bullfrog', 'Chinese edible frog', 'Beelzebufo', 'Scaphiophryne gottlebei', 'Hoplobatrachus tigerinus', 'Rhacophorus', 'Fire-bellied toad', 'Vietnamese mossy frog', 'Zhangixalus arboreus', 'Lepidobatrachus', 'Common tree frog', 'Golden mantella', 'Clawed frogs', 'Rhacophoridae', 'Pipidae', 'Pristimantis', 'Tomato frogs', 'Pelobates fuscus', 'Paedophryne amauensis', 'Gastric-brooding frog', 'Physalaemus gracilis', 'Smoky jungle frog', 'Mantella', 'Fire-bellied toads'],
    ['horses', 'Arabian horse', 'Friesian horse', 'Mustang', 'Thoroughbred', 'Shire horse', 'Appaloosa', 'American Quarter Horse', 'Clydesdale horse', 'Akhal-Teke', 'Galineers Cob', 'American Paint Horse', 'Dutch Warmblood', 'Haflinger', 'Mangalarga Marchador', 'Percheron', 'Turkoman horse', 'Breton horse', 'Shetland pony', 'Fjord horse', 'Criollo', 'Hanoverian horse', 'Morgan horse', 'Ardennais', 'Icelandic horse', 'Cob', 'Konik',
        'Belgian Draught', 'Lipizzan', 'Lusitano', 'Standardbred', 'Andalusian horse', 'Tennessee Walking Horse', 'Pure Spanish Breed', 'Falabella', 'Mongolian horse', 'Marwari horse', 'American Saddlebred', 'Knabstrupper', 'Trakehner', 'American Bashkir Curly', 'Missouri Fox Trotter', 'Belgian Warmblood', 'Noriker', 'Black Forest Horse', 'Holsteiner', 'Fell pony', 'Peruvian paso', 'Irish Sport Horse', 'Welsh Cob', 'Ferghana horse', 'Brumby'],
    ['deer', 'Moose', 'Reindeer', 'Roe deer', 'Elk', 'Red deer', 'European fallow deer', 'Alces', 'Muntjac', 'Sika deer', 'White-tailed deer', 'Mule deer', 'Chital', 'Sambar deer', 'Cervus', 'Roe deers', 'Water deer', "Reeves's muntjac", 'Barasingha', 'Indian hog deer',
        'Pudu', 'South Andean deer', 'Javan rusa', 'Irish elk', 'Cervinae', 'Tufted deer', 'Red brocket', "Père David's deer", "Eld's deer", 'Brown Brocket', 'Taruca', 'Brocket deer', 'Capreolinae', 'Dicrocerus elegans', 'Amazonian brown brocket', 'Croizetoceros'],
    ['birds', 'Budgerigar', 'Parrots', 'Chicken', 'Owl', 'Blue jay', 'Columbidae', 'Hummingbirds', 'Birds-of-paradise', 'Penguins', 'Toucans', 'Bluebirds', 'Common blackbird', 'Nyctibius', 'Finches', 'Cassowaries', 'Crows', 'European robin', 'Woodpeckers', 'Mallard', 'Falcon', 'Old World sparrows', 'House sparrow', 'Japanese bush warbler', 'Indian peafowl',
        'European goldfinch', 'Common ostrich', 'Barn swallow', 'Eurasian jay', 'Swallows', 'Warbling white-eye', 'Passerine', 'Atlantic canary', 'Common chaffinch', 'Herons', 'Cranes', 'Eurasian bullfinch', 'Emu', 'Pelican', 'Cuckoos', 'Common starling', 'Eurasian magpie', 'Kingfisher', 'Ostriches', 'Bald eagle', 'Swans', 'Common kingfisher', 'Stork', 'Eurasian hoopoe', 'Hornbill'],
    ['automobiles', 'SUV', 'Sedan', 'Crossover', 'Sports car', 'Hatchback', 'Convertible', 'Minivan', 'Coupe', 'Station Wagon', 'Vehicle', 'Compact car', 'Pickup truck',
        'Luxury', 'Roadster', 'Truck', 'Muscle car', 'Electric vehicle', 'Hybrid vehicle', 'Full-size car', 'Supercar', 'Microcar', 'Subcompact', 'Grand tourer', 'Limousine'],
    ['airplanes', 'Glider', 'Business jet', 'Very light jet', 'Cargo aircraft', 'Helicopter', 'Airbus A380', 'Boeing 747', 'Airliner', 'Boeing 787 Dreamliner', 'Boeing 777', 'Narrow-body aircraft',
        'Military aircraft', 'Boeing 767', 'Floatplane', 'Boeing 757', 'Airbus Beluga', 'Fighter aircraft', 'Amphibious', 'Concorde', 'Airbus A330', 'Aircraft', 'Light-sport aircraft', 'Boeing 747-8', 'Airbus A340'],
    ['dogs', 'German Shepherd', 'Bulldog', 'Labrador Retriever', 'Golden Retriever', 'French Bulldog', 'Siberian Husky', 'Alaskan Malamute', 'Poodle', 'Chihuahua', 'Border Collie', 'Afghan Hound', 'Airedale Terrier', 'Dachshund', 'Affenpinscher', 'Rottweiler', 'Bichon Frisé', 'Chow Chow', 'Australian Shepherd', 'Maltese dog', 'English Cocker Spaniel', 'Cavalier King Charles Spaniel', 'Pomeranian', 'Pembroke Welsh Corgi', 'American Eskimo Dog',
        'Anatolian Shepherd Dog', 'Yorkshire Terrier', 'Basset Hound', 'Basenji', 'Newfoundland dog', 'Havanese', 'Belgian Shepherd', 'Brittany', 'Sheltie', 'Bullmastiff', 'Boston Terrier', 'Cairn Terrier', 'Black Russian Terrier', 'Bedlington Terrier', 'American Pit Bull Terrier', 'Dobermann', 'Shiba Inu', 'Shih Tzu', 'Sarabi dog', 'Borzoi', 'Samoyed', 'American Bully', 'Jack Russell Terrier', 'Maltipoo', 'Goldendoodle', 'Dalmatian', 'Akita Inu'],
    ['cats', 'Siamese cat', 'British Shorthair', 'Maine Coon', 'Persian cat', 'Ragdoll', 'Sphynx cat', 'American Shorthair', 'Abyssinian', 'Exotic Shorthair', 'Scottish Fold', 'Burmese cat', 'Birman', 'Bombay cat', 'Siberian cat', 'Norwegian Forest cat', 'American Curl', 'Devon Rex', 'Russian Blue', 'Munchkin cat', 'American Bobtail', 'Oriental Shorthair', 'Balinese cat', 'Chartreux', 'Turkish Angora',
        'Japanese Bobtail', 'Manx Cat', 'American Wirehair', 'Ragamuffin', 'Somali cat', 'Egyptian Mau', 'Himalayan cat', 'Cornish Rex', 'Selkirk Rex', 'Korat', 'Ocicat', 'Singapura cat', 'Tonkinese cat', 'Turkish Van', 'British Longhair', 'LaPerm', 'Havana Brown', 'Chausie', 'Burmilla', 'Snowshoe cat', 'Sokoke', 'Toyger', 'Colorpoint Shorthair', 'Javanese cat', 'Australian Mist', 'Lykoi', 'Khao Manee'],
    ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]
]

# search for each keyword.
# store the images in a directory and links for images in a seperate files.
for list_of_queries in list_of_list_of_queries:
    for keyword in list_of_queries:

        # convert space seperated words to '-' seperated as unsplash uses this form
        keyword = "-".join(keyword.split(' '))

        # we will store a max of 50 images per keyword
        num_images = 50

        # store results in directory
        directory = 'unsplash_images'
        links_csv_path = directory+'links.csv'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # scrapper
        scrape_unsplash_images(keyword, num_images, directory, links_csv_path)
        time.sleep(10)
