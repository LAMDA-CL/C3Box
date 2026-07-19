import base64
import os
import time
from openai import OpenAI
import tqdm
import random

client = OpenAI(api_key="xxxxx")


def get_files_only(directory):
    if not os.path.exists(directory):
        return []
    all_entries = os.listdir(directory)
    # 过滤 .DS_Store 等系统文件
    files = [os.path.join(directory, entry) for entry in all_entries
             if os.path.isfile(os.path.join(directory, entry)) and not entry.startswith('.')]
    return files


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_image_description(image_path: str, category: str, output_path: str):
    image_name = os.path.basename(image_path)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            if image_name in f.read():
                return

    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    category_readable = category.replace("_", " ")

    prompt = f"""
You are an expert in fine-grained visual classification.
I will show you a low-resolution photo of a "{category_readable}".

Please provide a **single, concise, and detailed sentence** describing the visual appearance of this specific **{category_readable}** (or object) in the image.
Focus on: 
- The specific color and texture.
- The viewpoint or pose.
- Any distinct background environment.
- Distinguishing features visible despite the low resolution.

Constraint: Output ONLY the description sentence. Do not add "Here is the description" or quotes.
"""

    while True:
        try:
            response = client.chat.completions.create(
                model="YOUR_MODEL_NAME",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                },
                            },
                        ],
                    }
                ],
            )
            break
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            if "429" in str(e):
                time.sleep(10)
            else:
                time.sleep(3)

    description = response.choices[0].message.content
    description = description.replace("\n", " ")

    write_content = f"{image_name}\t{description}\n"

    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(write_content)
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    category_list = [
        "abbey",
        "airplane cabin",
        "airport terminal",
        "alley",
        "amphitheater",
        "amusement arcade",
        "amusement park",
        "anechoic chamber",
        "apartment building",
        "apse",
        "aquarium",
        "aqueduct",
        "arch",
        "archive",
        "arrival gate",
        "art gallery",
        "art school",
        "art studio",
        "assembly line",
        "athletic field",
        "atrium",
        "attic",
        "auditorium",
        "auto factory",
        "badlands",
        "badminton court",
        "baggage claim",
        "bakery",
        "balcony",
        "ball pit",
        "ballroom",
        "bamboo forest",
        "banquet hall",
        "bar",
        "barn",
        "barndoor",
        "baseball field",
        "basement",
        "basilica",
        "basketball court",
        "bathroom",
        "batters box",
        "bayou",
        "bazaar",
        "beach",
        "beauty salon",
        "bedroom",
        "berth",
        "biology laboratory",
        "bistro",
        "boardwalk",
        "boat deck",
        "boathouse",
        "bookstore",
        "booth",
        "botanical garden",
        "bow window",
        "bowling alley",
        "boxing ring",
        "brewery",
        "bridge",
        "building facade",
        "bullring",
        "burial chamber",
        "bus interior",
        "butchers shop",
        "butte",
        "cabin",
        "cafeteria",
        "campsite",
        "campus",
        "canal",
        "candy store",
        "canyon",
        "car interior",
        "carrousel",
        "casino",
        "castle",
        "catacomb",
        "cathedral",
        "cavern",
        "cemetery",
        "chalet",
        "cheese factory",
        "chemistry lab",
        "chicken coop",
        "childs room",
        "church",
        "classroom",
        "clean room",
        "cliff",
        "cloister",
        "closet",
        "clothing store",
        "coast",
        "cockpit",
        "coffee shop",
        "computer room",
        "conference center",
        "conference room",
        "construction site",
        "control room",
        "control tower",
        "corn field",
        "corral",
        "corridor",
        "cottage garden",
        "courthouse",
        "courtroom",
        "courtyard",
        "covered bridge",
        "creek",
        "crevasse",
        "crosswalk",
        "cubicle",
        "dam",
        "delicatessen",
        "dentists office",
        "desert",
        "diner",
        "dinette",
        "dining car",
        "dining room",
        "discotheque",
        "dock",
        "doorway",
        "dorm room",
        "driveway",
        "driving range",
        "drugstore",
        "electrical substation",
        "elevator",
        "elevator shaft",
        "engine room",
        "escalator",
        "excavation",
        "factory",
        "fairway",
        "fastfood restaurant",
        "field",
        "fire escape",
        "fire station",
        "firing range",
        "fishpond",
        "florist shop",
        "food court",
        "forest",
        "forest path",
        "forest road",
        "formal garden",
        "fountain",
        "galley",
        "game room",
        "garage",
        "garbage dump",
        "gas station",
        "gazebo",
        "general store",
        "gift shop",
        "golf course",
        "greenhouse",
        "gymnasium",
        "hangar",
        "harbor",
        "hayfield",
        "heliport",
        "herb garden",
        "highway",
        "hill",
        "home office",
        "hospital",
        "hospital room",
        "hot spring",
        "hot tub",
        "hotel",
        "hotel room",
        "house",
        "hunting lodge",
        "ice cream parlor",
        "ice floe",
        "ice shelf",
        "ice skating rink",
        "iceberg",
        "igloo",
        "industrial area",
        "inn",
        "islet",
        "jacuzzi",
        "jail",
        "jail cell",
        "jewelry shop",
        "kasbah",
        "kennel",
        "kindergarden classroom",
        "kitchen",
        "kitchenette",
        "labyrinth",
        "lake",
        "landfill",
        "landing deck",
        "laundromat",
        "lecture room",
        "library",
        "lido deck",
        "lift bridge",
        "lighthouse",
        "limousine interior",
        "living room",
        "lobby",
        "lock chamber",
        "locker room",
        "mansion",
        "manufactured home",
        "market",
        "marsh",
        "martial arts gym",
        "mausoleum",
        "medina",
        "moat",
        "monastery",
        "mosque",
        "motel",
        "mountain",
        "mountain snowy",
        "movie theater",
        "museum",
        "music store",
        "music studio",
        "nuclear power plant",
        "nursery",
        "oast house",
        "observatory",
        "ocean",
        "office",
        "office building",
        "oil refinery",
        "oilrig",
        "operating room",
        "orchard",
        "outhouse",
        "pagoda",
        "palace",
        "pantry",
        "park",
        "parking garage",
        "parking lot",
        "parlor",
        "pasture",
        "patio",
        "pavilion",
        "pharmacy",
        "phone booth",
        "physics laboratory",
        "picnic area",
        "pilothouse",
        "planetarium",
        "playground",
        "playroom",
        "plaza",
        "podium",
        "pond",
        "poolroom",
        "power plant",
        "promenade deck",
        "pub",
        "pulpit",
        "putting green",
        "racecourse",
        "raceway",
        "raft",
        "railroad track",
        "rainforest",
        "reception",
        "recreation room",
        "residential neighborhood",
        "restaurant",
        "restaurant kitchen",
        "restaurant patio",
        "rice paddy",
        "riding arena",
        "river",
        "rock arch",
        "rope bridge",
        "ruin",
        "runway",
        "sandbar",
        "sandbox",
        "sauna",
        "schoolhouse",
        "sea cliff",
        "server room",
        "shed",
        "shoe shop",
        "shopfront",
        "shopping mall",
        "shower",
        "skatepark",
        "ski lodge",
        "ski resort",
        "ski slope"
    ]
    formatted_category_list = [cate.replace(
        " ", "_") for cate in category_list]

    output_root = "<PATH>/generated_descriptions/"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for i, category in enumerate(formatted_category_list):
        print(
            f"********** Processing category {i+1}/{len(formatted_category_list)}: {category} **********")
        folder_path = f"<PATH>/train/{category}"
        out_file = f"{output_root}/{category.replace(" ", "_").replace("/", "_")}_descriptions.txt"
        if os.path.exists(out_file):
            print(
                f"Output file {out_file} already exists. Skipping category {category}.")
            continue

        file_list = get_files_only(folder_path)

        if not file_list:
            print(f"Warning: No files found for {category} in {folder_path}")
            continue

        for img_path in tqdm.tqdm(file_list):
            generate_image_description(img_path, category, out_file)
