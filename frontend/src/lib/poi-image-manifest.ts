export interface PoiImage {
  poiId: string;
  assetPath: string;
  originalSourceUrl: string;
  creator: string;
  license: string;
  licenseUrl: string;
  altText: string;
}

export const poiImageManifest = {
  "CF-001": {
    poiId: "CF-001",
    assetPath: "/images/poi/cf-001-old-fortress.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:The_Old_Fortress_and_the_Old_Town_of_Corfu_-_September_2017.jpg",
    creator: "Martin Falbisoner",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "The Old Fortress rising above the rooftops of Corfu's old town",
  },
  "CF-002": {
    poiId: "CF-002",
    assetPath: "/images/poi/cf-002-new-fortress.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:New_Venetian_Fortress,_Corfu_01.jpg",
    creator: "GualdimG",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "Stone ramparts and entrance at Corfu's New Venetian Fortress",
  },
  "CF-010": {
    poiId: "CF-010",
    assetPath: "/images/poi/cf-010-palace-st-michael-george.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:Palace_of_St._Michael_and_St._George,_Corfu_-_September_2017.jpg",
    creator: "Martin Falbisoner",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "The colonnaded Palace of St. Michael and St. George in Corfu",
  },
  "CF-014": {
    poiId: "CF-014",
    assetPath: "/images/poi/cf-014-municipal-art-gallery.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:Interior_of_Municipal_Gallery_of_Corfu_02.jpg",
    creator: "GualdimG",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "Paintings and display cases inside the Municipal Art Gallery of Corfu",
  },
  "CF-017": {
    poiId: "CF-017",
    assetPath: "/images/poi/cf-017-museum-asian-art.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:Museum_of_Asian_Art_of_Corfu_001.jpg",
    creator: "GualdimG",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "Asian artworks displayed in the Museum of Asian Art of Corfu",
  },
  "CF-021": {
    poiId: "CF-021",
    assetPath: "/images/poi/cf-021-achilleion.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:Korfu_(GR),_Gastouri,_Achilleion_--_2018_--_1313.jpg",
    creator: "Dietmar Rabich",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "Detail of a bronze classical figure at Achilleion Palace",
  },
  "CF-029": {
    poiId: "CF-029",
    assetPath: "/images/poi/cf-029-antivouniotissa.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:Antivouniotissa_Museum,_Corfu_04.jpg",
    creator: "GualdimG",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "The historic church that houses the Byzantine Museum of Antivouniotissa",
  },
  "CF-031": {
    poiId: "CF-031",
    assetPath: "/images/poi/cf-031-archaeological-museum.jpg",
    originalSourceUrl: "https://commons.wikimedia.org/wiki/File:Interior_of_the_Archaeological_Museum_of_Corfu,_with_the_West_Pediment_from_the_Temple_of_Artemis_in_Corfu_in_the_Background.jpg",
    creator: "Neoclassicism Enthusiast",
    license: "CC BY-SA 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
    altText: "The West Pediment from the Temple of Artemis on display in the Archaeological Museum of Corfu",
  },
} as const satisfies Record<string, PoiImage>;

export function getPoiImage(poiId: string): PoiImage | undefined {
  return poiImageManifest[poiId as keyof typeof poiImageManifest];
}
