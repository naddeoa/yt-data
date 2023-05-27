search_terms = [
    # "Minecraft tutorial",
    # "Fortnite gameplay",
    # "Apex Legends tips",
    # "Call of Duty multiplayer",
    # "League of Legends",
    # "Overwatch gameplay",
    # "GTA V walkthrough",
    # "The Legend of Zelda: Breath of the Wild review",
    # "Super Smash Bros. Ultimate tournaments",
    # "FIFA 22 career mode",
    # "Assassin's Creed Valhalla secrets",
    # "World of Warcraft raid guide",
    # "Valorant tips and tricks",
    # "Cyberpunk 2077 character customization",
    # "Rocket League gameplay",
    # "Pokemon Sword and Shield let's play",
    # "Battlefield V multiplayer",
    # "Destiny 2 raids",
    # "Fall Guys funny moments",
    # "Among Us strategy",
    # "The Witcher 3: Wild Hunt walkthrough",
    # "Resident Evil Village boss fights",
    # "Animal Crossing: New Horizons island design",
    # "Hearthstone deck building",
    # "Super Mario Odyssey speedrun",
    # "PlayerUnknown's Battlegrounds tactics",
    # "The Last of Us Part II gameplay",
    # "Dota 2 tournament highlights",
    # "Counter-Strike: Global Offensive weapon skins",
    # "Warframe beginner's guide",
    # "Madden NFL 22 gameplay",
    # "Borderlands 3 legendary weapons",
    # "Dead by Daylight survivor tips",
    # "Rainbow Six Siege ranked matches",
    # "Final Fantasy XIV class guide",
    # "Among Us live stream",
    # "The Elder Scrolls V: Skyrim mods",
    # "Fortnite dances",
    # "Mortal Kombat 11 fatalities",
    # "Farming Simulator 22 gameplay",
    # "Sea of Thieves treasure hunting",
    # "Pokémon GO raid battles",
    # "Valorant agent guide",
    # "Dark Souls III boss strategies",
    # "Street Fighter V combo tutorials",
    # "Stardew Valley farm layout ideas",
    # "Marvel's Spider-Man: Miles Morales gameplay",
    # "Rust base building",
    # "Tom Clancy's Ghost Recon Breakpoint stealth tactics",
    # "NBA 2K22 MyCareer mode",
    # "Resident Evil 2 remake walkthrough",
    # "Sekiro: Shadows Die Twice boss fights",
    # "ARK: Survival Evolved dinosaur taming",
    # "World of Warcraft: Shadowlands dungeon guide",
    # "Rainbow Six Siege operator tips",
    # "Grand Theft Auto Online heists",
    # "Fallout 4 settlement building",
    # "Red Dead Redemption 2 hunting tips",
    # "Phasmophobia gameplay",
    # "Assassin's Creed Odyssey exploration",
    # "The Sims 4 building",
    # "Super Smash Bros. Ultimate character analysis",
    # "Path of Exile build guide",
    # "Fortnite Battle Pass rewards",
    # "Monster Hunter: World weapon tutorials",
    # "Cuphead boss strategies",
    # "Dead Cells speedrun",
    # "Minecraft redstone creations",
    # "Tom Clancy's The Division 2 endgame content",
    # "Among Us impostor tactics",
    # "League of Legends champion spotlight",
    # "Genshin Impact character showcase",
    # "Apex Legends weapon tier list",
    # "The Legend of Zelda: Skyward Sword HD review",
    # "PlayerUnknown's Battlegrounds sniping tips",
    # "Madden NFL 22 franchise mode",
    # "No Man's Sky base building",
    # "The Witcher 3: Wild Hunt best armor",
    # "Ratchet & Clank: Rift Apart gameplay",
    # "FIFA 22 Ultimate Team tips",
    # "Fall Guys strategies",
    # "World of Warcraft leveling guide",
    #     "Fortnite item shop",
    # "Call of Duty Warzone loadout",
    # "Minecraft speedrun",
    # "Overwatch hero guides",
    # "Animal Crossing: New Horizons turnip prices",
    # "League of Legends eSports",
    # "Valorant ranked matches",
    # "GTA V cheat codes",
    # "Apex Legends character abilities",
    # "Assassin's Creed Valhalla hidden secrets",
    # "The Elder Scrolls Online PvP",
    # "FIFA 22 skill moves tutorial",
    # "Halo Infinite gameplay",
    # "Cyberpunk 2077 best builds",
    # "Monster Hunter Rise armor sets",
    # "Super Mario Maker 2 level showcase",
    # "Rainbow Six Siege operator loadouts",
    # "Among Us mod showcase",
    # "Pokémon Sword and Shield competitive battles",
    # "World of Warcraft gold farming",
    # "Rocket League trading",
    # "NBA 2K22 MyTeam mode",
    # "Resident Evil Village puzzle solutions",
    # "Mortal Kombat 11 character breakdowns",
    # "Fallout 76 base building tips",
    # "Sea of Thieves PvP encounters",
    # "The Sims 4 expansion pack reviews",
    # "Destiny 2 exotic weapon quests",
    # "Valorant map callouts",
    # "Final Fantasy VII Remake boss strategies",
    # "Stardew Valley fishing guide",
    # "Minecraft survival tips",
    # "Genshin Impact world exploration",
    # "Apex Legends sniping spots",
    # "The Legend of Zelda: Breath of the Wild secrets",
    # "FIFA 22 career mode wonderkids",
    # "Warframe prime warframes",
    # "Assassin's Creed Origins historical accuracy",
    # "The Witcher 3: Wild Hunt side quests",
    # "Fortnite creative mode showcases",
    # "Call of Duty zombies easter eggs",
    # "Pokémon Unite gameplay",
    # "Dota 2 hero spotlight",
    # "Super Smash Bros. Ultimate tier list",
    # "Tom Clancy's Ghost Recon Breakpoint weapon customization",
    # "Red Dead Redemption 2 legendary animals",
    # "Minecraft mods showcase",
    # "Sekiro: Shadows Die Twice speedrun",
    # "Phasmophobia ghost types",
    # "Borderlands 3 boss farming",
    # "ARK: Survival Evolved base defense strategies",
    # "Among Us venting tactics",
    #     "Super Mario 64 speedrun",
    # "The Legend of Zelda: Ocarina of Time walkthrough",
    # "Final Fantasy VII Remake review",
    # "Kingdom Hearts lore explained",
    # "Resident Evil 4 gameplay",
    # "Metal Gear Solid boss fights",
    # "Crash Bandicoot N. Sane Trilogy gameplay",
    # "Spyro the Dragon secrets",
    # "Pokemon Red and Blue nostalgia",
    # "GoldenEye 007 multiplayer",
    # "Gran Turismo 2 car tuning",
    # "Sonic the Hedgehog speedrun",
    # "Tomb Raider puzzle solutions",
    # "Street Fighter II combo tutorials",
    # "Super Smash Bros. Melee competitive scene",
    # "Jak and Daxter: The Precursor Legacy let's play",
    # "Tony Hawk's Pro Skater soundtrack",
    # "Silent Hill atmosphere analysis",
#     "Legend of Mana hidden treasures",
#     "Final Fantasy Tactics job guide",
#     "Kingdom Hearts boss strategies",
#     "Resident Evil 2 (1998) playthrough",
#     "Crash Team Racing shortcuts",
#     "Soulcalibur character creation",
#     "Metal Gear Solid storyline explained",
#     "Pokemon Stadium mini-games",
#     "Gran Turismo 3 A-Spec license tests",
#     "The Legend of Zelda: Majora's Mask secrets",
#     "Super Mario Sunshine 100% completion",
#     "Final Fantasy IX character analysis",
#     "Kingdom Hearts secret endings",
#     "Resident Evil Code: Veronica plot summary",
#     "Tekken 3 combo breakdowns",
#     "Metal Gear Solid 2: Sons of Liberty gameplay",
#     "Crash Nitro Kart multiplayer races",
#     "Shadow of the Colossus boss strategies",
#     "Resident Evil 3: Nemesis (1999) walkthrough",
#     "Final Fantasy X best sphere grid paths",
#     "Kingdom Hearts 2.5 Remix cutscene compilation",
#     "Mario Kart 64 shortcut guide",
#     "Tekken Tag Tournament character tier list",
#    "Super Mario World speedrun",
#     "The Legend of Zelda: A Link to the Past walkthrough",
#     "Final Fantasy VI best party composition",
#     "Kingdom Hearts 3 secret boss guide",
#     "Resident Evil (1996) mansion exploration",
#     "Chrono Trigger time travel mechanics",
#     "Crash Bandicoot 2: Cortex Strikes Back secrets",
#     "Metal Gear Solid 3: Snake Eater story analysis",
#     "Tekken 7 combo tutorials",
    # "Silent Hill 2 psychological horror analysis",
    # "Final Fantasy VIII Triple Triad strategies",
    # "Mario Kart: Double Dash!! character combinations",
    # "The Legend of Zelda: Wind Waker HD boss battles",
    # "Jak II open-world exploration",
    # "Shadow of the Colossus soundtrack appreciation",
    # "Resident Evil 4 weapon upgrades",
    # "Kingdom Hearts Birth by Sleep character backstories",
    # "Crash Bandicoot: Warped time trials",
    # "Final Fantasy VII original vs. remake comparison",
    # "Metal Gear Solid 4: Guns of the Patriots cutscene compilation",
    # "Tekken Tag Tournament 2 online tournaments",
    # "Donkey Kong Country secrets and hidden levels",
    # "Resident Evil 7: Biohazard VR experience",
    # "Super Mario RPG: Legend of the Seven Stars boss strategies",
    # "The Legend of Zelda: Skyward Sword HD controls guide",
    # "Final Fantasy XII hunts and rare game locations",
    # "Kingdom Hearts II final mix+ secret endings",
    # "Crash Bandicoot: The Wrath of Cortex gameplay",
    # "Metal Gear Solid V: The Phantom Pain base management tips",
    # "Tekken 5 character movelist breakdowns",
    # "Silent Hill 3 symbolism analysis",
    # "Final Fantasy Tactics Advance job combinations",
    # "Mario Party mini-games compilation",
    # "The Legend of Zelda: Twilight Princess HD hidden easter eggs",
    # "Resident Evil Remake puzzle solutions",
    # "Kingdom Hearts Re:Chain of Memories deck building strategies",
    # "Crash Bash multiplayer party mode",
    # "Metal Gear Solid Portable Ops storyline recap",
    # "Tekken 6 customization options",
    # "Donkey Kong Country 2: Diddy's Kong Quest speedrun",
    # "Resident Evil Zero inventory management tips",
    # "Final Fantasy XIII paradigm shifts explained",
    "Age of Empires II strategies",
    "StarCraft gameplay",
    "Command & Conquer: Red Alert mission walkthroughs",
    "Diablo II character builds",
    "Warcraft III: Reign of Chaos custom maps",
    "EverQuest classic raids",
    "Counter-Strike 1.6 competitive matches",
    "Baldur's Gate character creation",
    "Half-Life speedrun",
    "The Sims cheat codes",
    "Ultima Online player housing",
    "Deus Ex storyline analysis",
    "Fallout 2 tips and tricks",
    "Warhammer 40,000: Dawn of War tactics",
    "Diablo character classes compared",
    "Heroes of Might and Magic III strategy guide",
    "Commandos: Behind Enemy Lines mission strategies",
    "Quake III Arena frag compilation",
    "Star Wars: Knights of the Old Republic walkthrough",
    "Sid Meier's Civilization III diplomacy tips",
    "Red Alert 2 Soviet campaign playthrough",
    "EverQuest II questing guide",
    "Age of Mythology god powers explained",
    "Diablo II: Lord of Destruction rune words",
    "Warcraft II: Tides of Darkness Orc campaign walkthrough",
    "The Elder Scrolls III: Morrowind mod showcase",
    "Max Payne bullet time mechanics",
    "RollerCoaster Tycoon park design",
    "Warcraft III: The Frozen Throne competitive strategies",
    "Command & Conquer: Generals base building tips",
    "Neverwinter Nights character progression",
    "Counter-Strike: Source weapon spray patterns",
    "Age of Empires civilization overviews",
    "Diablo III endgame content",
    "Guild Wars elite areas",
    "Red Alert 2 Yuri's Revenge expansion tactics",
    "Fallout Tactics squad management",
    "Warcraft III custom campaigns",
    "The Sims 2 expansion pack reviews",
    "Star Wars Galaxies crafting system",
    "Command & Conquer: Red Alert 2 mods",
    "Deus Ex: Invisible War plot analysis",
    "Myst puzzle solutions",
    "Age of Mythology: The Titans campaign playthrough",
    "SimCity 3000 city planning tips",
    "Heroes of Might and Magic IV hero builds",
    "Warcraft II: Beyond the Dark Portal human campaign guide",
    "The Elder Scrolls IV: Oblivion vampire guide",
    "Max Payne 2: The Fall of Max Payne story recap",
    "Rise of Nations resource management strategies",
]