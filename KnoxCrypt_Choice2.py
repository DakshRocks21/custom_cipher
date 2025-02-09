"Written by Daksh"

"""
Stronger KnoxCrypt. 
Roating Grid + Knapsack Cipher with BB84 Quantum Key Exchange (w one time pad)
"""

import random
import numpy as np
from math import gcd

class BB84:
    """
    Adapated from BB84_KeyExchange.py
    """
    def __init__(self, bit_length, toggle_eve=False, transmission_noise=0.0, measurement_error=0.0, overide=False):
        self.bit_length = bit_length
        self.toggle_eve = toggle_eve
        self.transmission_noise = transmission_noise
        self.measurement_error = measurement_error
        self.overide = overide

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

        # Take a random sample for error checking.
        sample_size = min(10, len(alice_key)) 
        if sample_size > 0:
            sample_indices = random.sample(range(len(alice_key)), sample_size)
            alice_sample = [alice_key[i] for i in sample_indices]
            bob_sample = [bob_key[i] for i in sample_indices]
            error_rate = sum(1 for a, b in zip(alice_sample, bob_sample) if a != b) / sample_size
        else:
            error_rate = 0

        print(f"Alice's Key      : {alice_key}")
        print(f"Bob's Key        : {bob_key}")
        print(f"Error Rate       : {error_rate}")

        if alice_key == bob_key and len(alice_key) > 0:
            print("\n[BB84] Key Exchange Successful!\n")
            return alice_key, bob_key
        else:
            if self.overide:
                print("\n[BB84] Key Exchange Failed. Overriding...\n")
                return alice_key, bob_key
            print("\n[BB84] Key Exchange Failed. Retrying...\n")
            return "", ""

###############################
# Rotating Grid (Cardan Grille) 
###############################
class RotatingGridCipher:
    """
    Adapted from https://py.checkio.org/en/mission/rotating-grille-cipher/share/980ce1df57b98e4bdf955172c0b66f74/
    """
    def __init__(self, size=4):
        self.size = size  

    def generate_grid(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        num_holes = (self.size * self.size) // 4  
        available = set(range(self.size * self.size))
        holes = []
        while len(holes) < num_holes:
            pos = random.choice(list(available))
            i = pos // self.size
            j = pos % self.size
            positions = {pos,
                         j * self.size + (self.size - 1 - i), 
                         (self.size - 1 - i) * self.size + (self.size - 1 - j),
                         (self.size - 1 - j) * self.size + i}
            if positions.issubset(available):
                holes.append(pos)
                available -= positions
        for pos in holes:
            i = pos // self.size
            j = pos % self.size
            grid[i][j] = 1
        return grid

    def rotate_grid(self, grid):
        return np.rot90(grid, -1)

    def encrypt(self, plaintext, grid):
        """
        Encrypt one block (of length size^2) using the Cardan grille.
        The plaintext is written into the holes over four rotations.
        """
        block_size = self.size ** 2
        plaintext = plaintext.ljust(block_size, 'X')
        ciphertext = [''] * block_size
        index = 0
        temp_grid = grid.copy()
        for _ in range(4):
            for row in range(self.size):
                for col in range(self.size):
                    if temp_grid[row, col] == 1:
                        ciphertext[row * self.size + col] = plaintext[index]
                        index += 1
            temp_grid = self.rotate_grid(temp_grid)
        # Fill any remaining blanks with padding.
        for k in range(block_size):
            if ciphertext[k] == '':
                ciphertext[k] = 'X'
        return ''.join(ciphertext)

    def decrypt(self, ciphertext, grid):
        """
        Decrypt a block that was encrypted with the Cardan grille.
        The holes (in order of rotations) reveal the plaintext.
        """
        block_size = self.size ** 2
        plaintext_chars = [''] * block_size
        index = 0
        temp_grid = grid.copy()
        for _ in range(4):
            for row in range(self.size):
                for col in range(self.size):
                    if temp_grid[row, col] == 1:
                        plaintext_chars[index] = ciphertext[row * self.size + col]
                        index += 1
            temp_grid = self.rotate_grid(temp_grid)
        result = ''.join(plaintext_chars)
        return result.rstrip('X')

###############################
# Knapsack Cryptosystem (Merkle–Hellman style)
###############################
def generate_super_increasing_sequence(n):
    seq = [random.randint(1, 100)]
    for _ in range(1, n):
        seq.append(sum(seq) + random.randint(1, 100))
    return seq

class Knapsack:
    def __init__(self, name, bb84_key):
        """
        The secret key (bb84_key) is used during key generation.
        """
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
        cipher_values = []
        for binary in plaintext_binary:
            cipher_sum = sum(int(bit) * recipient_public_key[i] for i, bit in enumerate(binary))
            cipher_values.append(cipher_sum)
        return cipher_values

    def decrypt(self, ciphertext):
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
        return [format(ord(c), '08b') for c in text]

    def binary_to_text(self, binary_list):
        try:
            return ''.join(chr(int(b, 2)) for b in binary_list)
        except Exception as e:
            return "Decryption Error: " + str(e)

###############################
# Key Exchanger
###############################
class KeyExchanger:
    def __init__(self, bb84, grid_cipher):
        self.bb84 = bb84
        self.grid_cipher = grid_cipher

    def exchange_grid(self):
        """
        Used some AI to make this code better and to fix bugs.
        """
        
        # Step 1: Alice generates a valid grid.
        alice_grid = self.grid_cipher.generate_grid()
        size = self.grid_cipher.size
        grid_binary = ''.join(''.join(str(cell) for cell in row) for row in alice_grid)
        print(f"\n[KeyExchanger] Alice's grid (binary): {grid_binary}")

        alice_bb84_key, bob_bb84_key = self.bb84.simulate()
        attempt = 1
        # Ensure that keys are non-empty, match, and have sufficient length
        while (alice_bb84_key == "" or bob_bb84_key == "" or
               alice_bb84_key != bob_bb84_key or
               len(alice_bb84_key) < len(grid_binary)):
            print(f"[KeyExchanger] Attempt {attempt}: BB84 key generation failed or key length too short. Retrying...")
            alice_bb84_key, bob_bb84_key = self.bb84.simulate()
            attempt += 1

        print("[KeyExchanger] BB84 key generation succeeded and key length is sufficient.")

        # Step 3: Encrypt the grid using the shared BB84 key as a one-time pad.
        key_segment = alice_bb84_key[:len(grid_binary)]
        encrypted_grid = ''.join('1' if gb != kb else '0' for gb, kb in zip(grid_binary, key_segment))
        print(f"[KeyExchanger] Encrypted grid (via XOR with BB84 key segment): {encrypted_grid}")

        # Simulate sending the encrypted grid over a classical channel...
        decrypted_grid = ''.join('1' if eb != kb else '0' for eb, kb in zip(encrypted_grid, bob_bb84_key[:len(grid_binary)]))
        print(f"[KeyExchanger] Bob's decrypted grid (binary): {decrypted_grid}")

        # Reconstruct Bob's grid from the decrypted binary string.
        bob_grid = np.array([[int(decrypted_grid[i * size + j]) for j in range(size)] for i in range(size)])
        print("[KeyExchanger] Grid successfully exchanged via proper BB84.\n")
        return alice_grid, bob_grid


###############################
# KNOXCRYPT CHOICE 2
###############################
class KnoxCrypt:
    def __init__(self, knapsack_instance, grid, grid_cipher):
        self.knapsack = knapsack_instance  
        self.grid = grid                    
        self.grid_cipher = grid_cipher
        self.block_size = grid_cipher.size ** 2  

    def encrypt(self, plaintext, recipient_public_key):
        # --- Knapsack Encryption ---
        binary_list = self.knapsack.text_to_binary(plaintext)
        knapsack_cipher = self.knapsack.encrypt(binary_list, recipient_public_key)
        # Convert the list of integers into a comma-separated string.
        intermediate = ','.join(str(num) for num in knapsack_cipher)
        # Pad the intermediate string to a multiple of block_size using 'X'
        pad_length = ((len(intermediate) + self.block_size - 1) // self.block_size) * self.block_size
        padded = intermediate.ljust(pad_length, 'X')
        # Break into fixed-size blocks.
        blocks = [padded[i:i+self.block_size] for i in range(0, len(padded), self.block_size)]
        # Encrypt each block using the grid cipher.
        encrypted_blocks = [self.grid_cipher.encrypt(block, self.grid) for block in blocks]
        # Join blocks (they have fixed size, so no delimiter is needed).
        final_ciphertext = ''.join(encrypted_blocks)
        return final_ciphertext

    def decrypt(self, ciphertext):
        # Break ciphertext into blocks of block_size.
        nblocks = len(ciphertext) // self.block_size
        blocks = [ciphertext[i*self.block_size:(i+1)*self.block_size] for i in range(nblocks)]
        decrypted_blocks = [self.grid_cipher.decrypt(block, self.grid) for block in blocks]
        intermediate = ''.join(decrypted_blocks)
        # Remove the padding added during encryption.
        intermediate = intermediate.rstrip('X')
        # Convert the intermediate string back into knapsack ciphertext integers.
        parts = intermediate.split(',')
        try:
            knapsack_cipher = [int(part) for part in parts if part != '']
        except Exception as e:
            return "Decryption Error: invalid format."
        binary_list = self.knapsack.decrypt(knapsack_cipher)
        plaintext = self.knapsack.binary_to_text(binary_list)
        return plaintext

###############################
# Two-Way Hybrid Protocol
###############################
class TwoWayHybridProtocol:
    def __init__(self, bb84_bit_length=32, toggle_eve=False, transmission_noise=0.0, measurement_error=0.0):
        # Initialize the improved BB84 and the grid cipher.
        self.bb84 = BB84(bit_length=bb84_bit_length, toggle_eve=toggle_eve,
                           transmission_noise=transmission_noise, measurement_error=measurement_error)
        self.grid_cipher = RotatingGridCipher(size=4)
        
        # Use the KeyExchanger to securely transfer the grid using BB84.
        self.key_exchanger = KeyExchanger(self.bb84, self.grid_cipher)
        self.alice_grid, self.bob_grid = self.key_exchanger.exchange_grid()
        if self.alice_grid is None or self.bob_grid is None:
            raise Exception("Grid exchange failed due to BB84 key mismatch.")

        grid_binary = ''.join(''.join(str(cell) for cell in row) for row in self.alice_grid)
        secret_for_knapsack = grid_binary[:8]

        self.alice_knapsack = Knapsack("Alice", secret_for_knapsack)
        self.bob_knapsack = Knapsack("Bob", secret_for_knapsack)

        self.alice_hybrid = KnoxCrypt(self.alice_knapsack, self.alice_grid, self.grid_cipher)
        self.bob_hybrid = KnoxCrypt(self.bob_knapsack, self.bob_grid, self.grid_cipher)

    def run(self):
        # --- Alice sends a message to Bob ---
        alice_message = "Hello, Bob! This is a Hybrid Cipher message."
        print("Alice sends:", alice_message)
        ciphertext = self.alice_hybrid.encrypt(alice_message, self.bob_knapsack.public_key)
        print("Ciphertext (Alice->Bob):", ciphertext)
        decrypted_by_bob = self.bob_hybrid.decrypt(ciphertext)
        print("Bob receives:", decrypted_by_bob)

        # --- Bob sends a message to Alice ---
        bob_message = "Hi, Alice! Message received loud and clear."
        print("\nBob sends:", bob_message)
        ciphertext2 = self.bob_hybrid.encrypt(bob_message, self.alice_knapsack.public_key)
        print("Ciphertext (Bob->Alice):", ciphertext2)
        decrypted_by_alice = self.alice_hybrid.decrypt(ciphertext2)
        print("Alice receives:", decrypted_by_alice)


protocol = TwoWayHybridProtocol(bb84_bit_length=32, toggle_eve=False, transmission_noise=0.00, measurement_error=0.00)
protocol.run()
