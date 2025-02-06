import random
from math import gcd


class BB84:
    def __init__(self, bit_length, toggle_eve=False, transmission_noise=0.0, measurement_error=0.0, overide=False):
        self.bit_length = bit_length
        self.toggle_eve = toggle_eve
        self.transmission_noise = transmission_noise
        self.measurement_error = measurement_error
        self.overide = overide
        self.key = None

    def generate_random_bits(self, length):
        return [random.randint(0, 1) for _ in range(length)]

    def compare_bases(self, alice_bases, bob_bases):
        return [i for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]

    def extract_key(self, bits, matching_indices):
        return ''.join([str(bits[i]) for i in matching_indices])

    def simulate(self):
        print("\n[BB84] Simulating Quantum Key Exchange...\n")
        alice_bits = self.generate_random_bits(self.bit_length)
        alice_bases = self.generate_random_bits(self.bit_length)

        if self.toggle_eve:
            eve_bases = self.generate_random_bits(self.bit_length)
            eve_measured_bits = [
                alice_bits[i] if alice_bases[i] == eve_bases[i] else random.randint(0, 1)
                for i in range(self.bit_length)
            ]
            bob_bits_received = [
                eve_measured_bits[i] if random.random() > self.transmission_noise else 1 - eve_measured_bits[i]
                for i in range(self.bit_length)
            ]
        else:
            bob_bits_received = [
                alice_bits[i] if random.random() > self.transmission_noise else 1 - alice_bits[i]
                for i in range(self.bit_length)
            ]

        bob_bases = self.generate_random_bits(self.bit_length)
        bob_measured_bits = [
            (bob_bits_received[i] if alice_bases[i] == bob_bases[i] else random.randint(0, 1))
            if random.random() > self.measurement_error else 1 - bob_bits_received[i]
            for i in range(self.bit_length)
        ]

        matching_indices = self.compare_bases(alice_bases, bob_bases)
        alice_key = self.extract_key(alice_bits, matching_indices)
        bob_key = self.extract_key(bob_measured_bits, matching_indices)

        sample_size = min(10, len(alice_key)) 
        sample_indices = random.sample(range(len(alice_key)), sample_size)
        alice_sample = [alice_key[i] for i in sample_indices]
        bob_sample = [bob_key[i] for i in sample_indices]

        error_rate = sum([1 for a, b in zip(alice_sample, bob_sample) if a != b]) / sample_size

        print(f"Alice's Key      : {alice_key}")
        print(f"Bob's Key        : {bob_key}")
        print(f"Error Rate       : {error_rate}")

        if alice_key == bob_key:
            print("\n[BB84] Key Exchange Successful!\n")
            return alice_key, bob_key
        else:
            if self.overide:
                print("\n[BB84] Key Exchange Failed. Overriding...\n")
                return alice_key, bob_key
            
            print("\n[BB84] Key Exchange Failed. Retrying...\n")
            return "", ""


def generate_super_increasing_sequence(n):
    knapsack = [random.randint(1, 100)]
    for _ in range(1, n):
        knapsack.append(sum(knapsack) + random.randint(1, 100))
    return knapsack


class Knapsack:
    def __init__(self, name, bb84_key):
        self.name = name
        self.secret_key = bb84_key
        self.public_key, self.private_key, self.q, self.r = self.generate_keys(len(bb84_key))

    def generate_keys(self, length):
        private_key = generate_super_increasing_sequence(length)
        q = sum(private_key) + random.randint(1, 100)
        r = random.randint(2, q - 1)
        while gcd(r, q) != 1:
            r = random.randint(2, q - 1)
        public_key = [(r * k) % q for k in private_key]
        return public_key, private_key, q, r

    def encrypt(self, plaintext_binary, recipient_public_key):
        xor_output = self.binary_xor(plaintext_binary)
        ciphertext = self.knapsack_encrypt_with_public_key(xor_output, recipient_public_key)
        return ciphertext

    def decrypt(self, ciphertext):
        decrypted_binary = self.knapsack_decrypt_with_private_key(ciphertext)
        plaintext_binary = self.binary_xor(decrypted_binary)
        return plaintext_binary

    def binary_xor(self, binary_list):
        return [format(int(b, 2) ^ int(self.secret_key, 2), f'0{len(self.secret_key)}b') for b in binary_list]

    def knapsack_encrypt_with_public_key(self, binary_list, public_key):
        cipher_values = []
        for binary in binary_list:
            cipher_sum = sum(int(bit) * public_key[i] for i, bit in enumerate(binary))
            cipher_values.append(cipher_sum)
        return cipher_values

    def knapsack_decrypt_with_private_key(self, ciphertext):
        r_inverse = pow(self.r, -1, self.q)
        binary_list = []
        for cipher in ciphertext:
            plain_sum = (cipher * r_inverse) % self.q
            binary_chunk = ['0'] * len(self.private_key)
            for i in range(len(self.private_key) - 1, -1, -1):
                if plain_sum >= self.private_key[i]:
                    plain_sum -= self.private_key[i]
                    binary_chunk[i] = '1'
            binary_list.append(''.join(binary_chunk))
        return binary_list

    def text_to_binary(self, text):
        return [format(ord(char), '08b') for char in text]

    def binary_to_text(self, binary_list):
        try: 
            return ''.join([chr(int(b, 2)) for b in binary_list])
        except ValueError:
            return "Decryption Error, values outside ASCII range"
        except OverflowError:
            return "Decryption Error: We have hit the limits of a C integer"


class TwoWayKnapsackWithBB84:
    def __init__(self, bit_length=16, toggle_eve=False, **kwargs):
        self.bb84 = BB84(bit_length, toggle_eve, **kwargs)
        self.kwargs = kwargs
        self.alice = None
        self.bob = None

    def run(self):
        print("\nWelcome to the Two-Way Knapsack with BB84 Protocol\n")
        print("[+] Initializing BB84 Quantum Key Exchange...\n")

        alice_key, bob_key = "",""
        
        count = 1
        
        while (len(alice_key) < 8 or len(bob_key) < 8):
            self.bb84 = BB84(bit_length=self.bb84.bit_length, toggle_eve=self.bb84.toggle_eve, **self.kwargs)
            alice_key, bob_key = self.bb84.simulate()
            
            if alice_key == "" or bob_key == "":
                count += 1
                continue
            else:
                if len(alice_key) >= 8 and len(bob_key) >= 8:
                    print("[+] Took", count, "attempts to generate a valid key\n\n")
                    break    
            
            print("Error: BB84 key length too short. Retrying...\n")
                
        
        self.alice = Knapsack("Alice", alice_key)
        self.bob = Knapsack("Bob", bob_key)
        
        # print("Alice's Public Key:", self.alice.public_key)
        # print("Bob's Public Key  :", self.bob.public_key)
        
        # print("Alice's Private Key:", self.alice.private_key)
        # print("Bob's Private Key  :", self.bob.private_key)


        alice_message = "Hello, Bob!"
        print(f"\n[+] Alice sends a message to Bob: {alice_message}\n")
        alice_binary = self.alice.text_to_binary(alice_message)
        alice_ciphertext = self.alice.encrypt(alice_binary, self.bob.public_key)
        print(f"Alice's Encrypted Message: {alice_ciphertext}\n")


        bob_received_binary = self.bob.decrypt(alice_ciphertext)
        bob_received_message = self.bob.binary_to_text(bob_received_binary)
        print(f"Bob Decrypts and Receives: {bob_received_message}\n")


        bob_message = "Hello, Alice!"
        print(f"\n[+] Bob sends a response to Alice: {bob_message}\n")
        bob_binary = self.bob.text_to_binary(bob_message)
        bob_ciphertext = self.bob.encrypt(bob_binary, self.alice.public_key)
        print(f"Bob's Encrypted Message: {bob_ciphertext}\n")


        alice_received_binary = self.alice.decrypt(bob_ciphertext)
        alice_received_message = self.alice.binary_to_text(alice_received_binary)
        print(f"Alice Decrypts and Receives: {alice_received_message}\n")

system = TwoWayKnapsackWithBB84(bit_length=32, toggle_eve=False)
system.run()
