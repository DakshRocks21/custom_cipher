import random
from math import gcd
import os

def generate_super_increasing_sequence(n):
    knapsack = [random.randint(1, 100)]
    for _ in range(1, n):
        knapsack.append(sum(knapsack) + random.randint(1, 100))
    return knapsack

class Knapsack:
    def __init__(self, name):
        self.name = name
        self.public_key, self.private_key, self.q, self.r = self.generate_keys()

    def generate_keys(self):
        private_key = generate_super_increasing_sequence(8) 
        q = sum(private_key) + random.randint(1, 100)
        r = random.randint(2, q - 1)
        
        while gcd(r, q) != 1:
            r = random.randint(2, q - 1)

        public_key = [(r * k) % q for k in private_key]
        return public_key, private_key, q, r

    def encrypt(self, plaintext, recipient_public_key):
        binary_text = self.text_to_binary(plaintext)
        ciphertext = [sum(int(bit) * recipient_public_key[i] for i, bit in enumerate(binary_char)) for binary_char in binary_text]
        return ciphertext

    def decrypt(self, ciphertext):
        r_inverse = pow(self.r, -1, self.q)
        decrypted_binary = []
        
        for cipher in ciphertext:
            plain_sum = (cipher * r_inverse) % self.q
            binary_chunk = ['0'] * len(self.private_key)

            for i in range(len(self.private_key) - 1, -1, -1):  
                if plain_sum >= self.private_key[i]:
                    plain_sum -= self.private_key[i]
                    binary_chunk[i] = '1'

            decrypted_binary.append(''.join(binary_chunk))
        
        return self.binary_to_text(decrypted_binary)

    def text_to_binary(self, text):
        return [format(ord(char), '08b') for char in text]

    def binary_to_text(self, binary_list):
        try:
            return ''.join([chr(int(b, 2)) for b in binary_list])
        except ValueError:
            return "Decryption Error: Invalid ASCII values."

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


clear_screen()
print("=== Secure Knapsack Cryptosystem Chat ===")
print("\n[+] Generating cryptographic keys for Alice and Bob.", end="")

alice = Knapsack("Alice")
print("\n[+] Alice's keys have been generated successfully.")
print("     [+] Alice's public key:", alice.public_key)
print("     [+] Alice's private key:", alice.private_key)

bob = Knapsack("Bob")
print("\n[+] Bob's keys have been generated successfully.")
print("     [+] Bob's public key:", bob.public_key)
print("     [+] Bob's private key:", bob.private_key)

print("\n[+] Alice and Bob are now ready to communicate securely.")
input("\nPress Enter to continue to chat...")

while True:
    clear_screen()
    print("\n=== Secure Knapsack Cryptosystem Chat ===")
    print("[1] Alice sends a message")
    print("[2] Bob sends a message")
    print("[3] Exit")
    
    choice = input("\nChoose an option: ")

    if choice == "1":
        message = input("\n[+] Alice, enter your message: ")
        print(f"\n[+] Encrypting message using Bob's public key...")
        encrypted_message = alice.encrypt(message, bob.public_key)
        print(f"Alice's Encrypted Message: {encrypted_message}")

        print("\n[+] Decrypting message using Bob's private key...")
        decrypted_message = bob.decrypt(encrypted_message)
        print(f"[Bob receives]: {decrypted_message}")


    elif choice == "2" :
        message = input("\n[+] Bob, enter your message: ")
        print(f"\n[+] Encrypting message using Alice's public key...")
        encrypted_message = bob.encrypt(message, alice.public_key)
        print(f"Bob's Encrypted Message: {encrypted_message}")

        print("\n[+] Decrypting message using Alice's private key...")
        decrypted_message = alice.decrypt(encrypted_message)
        print(f"[Alice receives]: {decrypted_message}")

    elif choice == "3":
        print("\n[+] Exiting secure chat.")
        break

    else:
        print("\n[!] Invalid choice. Please try again.")

    input("\nPress Enter to continue...")
