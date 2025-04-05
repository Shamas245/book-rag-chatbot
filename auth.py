import json

class AuthManager:
    @staticmethod
    def login(username: str, password: str) -> bool:
        try:
            with open("users.json", "r") as f:
                users = json.load(f)
            return username in users and users[username] == password
        except FileNotFoundError:
            return False

    @staticmethod
    def register(username: str, password: str) -> bool:
        try:
            with open("users.json", "r") as f:
                users = json.load(f)
        except FileNotFoundError:
            users = {}
            
        if username in users:
            return False
        
        users[username] = password
        with open("users.json", "w") as f:
            json.dump(users, f)
        return True