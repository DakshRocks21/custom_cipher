import random
import numpy as np
from math import gcd

###############################
# BB84 Simulation
###############################
class BB84:
    def __init__(self, bit_length, toggle_eve=False, transmission_noise=0.0, measurement_error=0.0, override=False):
        self.bit_length = bit_length
        self.toggle_eve = toggle_eve
        self.transmission_noise = transmission_noise
        self.measurement_error = measurement_error
        self.override = override

    def simulate(self, secret=None):
        """
        If a secret (a binary string) is provided, then simulate an ideal BB84 transfer.
        Otherwise, generate a random key and simulate the BB84 protocol.
        
        Added a secret parameter just for the hybrid design.
        """
        if secret is not None:
            # Use the provided secret as Alice's key; assume Bob receives it perfectly.
            # TODO: Add noise to simulate real-world conditions.
            # TODO: Implement Eve's interception and retransmission.
            alice_key = secret
            bob_key = secret
            print("[BB84] Transferred secret via BB84 channel.")
            return alice_key, bob_key
        else:
            # Default simulation (unused in this hybrid design, copied from previous code).
            alice_bits = [random.randint(0, 1) for _ in range(self.bit_length)]
            alice_bases = [random.randint(0, 1) for _ in range(self.bit_length)]
            bob_bases = [random.randint(0, 1) for _ in range(self.bit_length)]
            bob_measured = [alice_bits[i] if alice_bases[i] == bob_bases[i] else random.randint(0,1)
                            for i in range(self.bit_length)]
            matching = [i for i in range(self.bit_length) if alice_bases[i] == bob_bases[i]]
            alice_key = ''.join(str(alice_bits[i]) for i in matching)
            bob_key = ''.join(str(bob_measured[i]) for i in matching)
            if alice_key == bob_key:
                print("[BB84] Key Exchange Successful!")
                return alice_key, bob_key
            else:
                print("[BB84] Key Exchange Failed.")
                return "", ""

###############################
# Rotating Grid (Cardan Grille) Cipher
###############################
class RotatingGridCipher:
    def __init__(self, size=4):
        self.size = size  # For a 4x4 grid

    def generate_grid(self):
        """
        Generates a valid Cardan grille for a square of given size.
        For a 4×4 grid, exactly (16//4)=4 ones are placed so that, over 4 rotations,
        every cell is covered exactly once.
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        num_holes = (self.size * self.size) // 4  # e.g., 4 holes for 4x4 grid
        available = set(range(self.size * self.size))
        holes = []
        while len(holes) < num_holes:
            pos = random.choice(list(available))
            i = pos // self.size
            j = pos % self.size
            # Determine all four rotational positions.
            positions = set()
            positions.add(pos)  # 0°: (i, j)
            pos1 = j * self.size + (self.size - 1 - i)        # 90°: (j, size-1-i)
            positions.add(pos1)
            pos2 = (self.size - 1 - i) * self.size + (self.size - 1 - j)  # 180°: (size-1-i, size-1-j)
            positions.add(pos2)
            pos3 = (self.size - 1 - j) * self.size + i          # 270°: (size-1-j, i)
            positions.add(pos3)
            if positions.issubset(available):
                holes.append(pos)
                available -= positions
        # Place holes in the grid.
        for pos in holes:
            i = pos // self.size
            j = pos % self.size
            grid[i][j] = 1
        return grid

    def rotate_grid(self, grid):
        return np.rot90(grid, -1)

    def encrypt(self, plaintext, grid):
        """
        Encrypt a block of text using the Cardan grille.
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
        for k in range(block_size):
            if ciphertext[k] == '':
                ciphertext[k] = 'X'
        return ''.join(ciphertext)

    def decrypt(self, ciphertext, grid):
        """
        Decrypt a block of text that was encrypted using the Cardan grille.
        The decryption recovers characters in the order defined by the grille rotations.
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
        The secret (bb84_key) is used in key generation.
        Here, we expect bb84_key to be 8 bits.
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
        """
        Encrypt a list of binary strings (each representing 8 bits).
        Returns a list of integers.
        """
        cipher_values = []
        for binary in plaintext_binary:
            cipher_sum = sum(int(bit) * recipient_public_key[i] for i, bit in enumerate(binary))
            cipher_values.append(cipher_sum)
        return cipher_values

    def decrypt(self, ciphertext):
        """
        Decrypts a list of integers and returns a list of binary strings.
        """
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
# Key Exchanger (BB84 transferring the grid)
###############################
class KeyExchanger:
    def __init__(self, bb84, grid_cipher):
        self.bb84 = bb84
        self.grid_cipher = grid_cipher

    def exchange_grid(self):
        # Step 1: Alice generates a valid grid.
        alice_grid = self.grid_cipher.generate_grid()
        # Convert the grid to a binary string (row-major order).
        size = self.grid_cipher.size
        grid_binary = ''.join(''.join(str(cell) for cell in row) for row in alice_grid)
        print(f"\n[KeyExchanger] Alice's grid (binary): {grid_binary}")
        # Step 2: Use BB84 to transfer this binary string.
        alice_key, bob_key = self.bb84.simulate(secret=grid_binary)
        # Reconstruct Bob's grid from the received binary string.
        bob_grid = np.array([[int(bob_key[i*size + j]) for j in range(size)] for i in range(size)])
        print("[KeyExchanger] Grid successfully exchanged via BB84.\n")
        return alice_grid, bob_grid

###############################
# Hybrid Cipher (Knapsack + Rotating Grid)
###############################
class HybridCipher:
    def __init__(self, knapsack_instance, grid, grid_cipher):
        self.knapsack = knapsack_instance  # Instance of Knapsack for the current party.
        self.grid = grid                    # Shared grid (from BB84 exchange).
        self.grid_cipher = grid_cipher
        self.block_size = grid_cipher.size ** 2  # 16 for a 4x4 grid

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
    def __init__(self, bit_length=16, toggle_eve=False):
        # Initialize BB84 and the grid cipher.
        self.bb84 = BB84(bit_length, toggle_eve)
        self.grid_cipher = RotatingGridCipher(size=4)
        # Use KeyExchanger to securely transfer the grid.
        self.key_exchanger = KeyExchanger(self.bb84, self.grid_cipher)
        self.alice_grid, self.bob_grid = self.key_exchanger.exchange_grid()

        # The shared grid's binary representation (from a 4x4 grid, 16 bits) is produced.
        # For Knapsack we use only 8 bits (to match the 8-bit text conversion).
        size = self.grid_cipher.size
        grid_binary = ''.join(''.join(str(cell) for cell in row) for row in self.alice_grid)
        secret_for_knapsack = grid_binary[:8]  # Use only the first 8 bits.
        # Initialize Knapsack for both parties with the secret.
        self.alice_knapsack = Knapsack("Alice", secret_for_knapsack)
        self.bob_knapsack = Knapsack("Bob", secret_for_knapsack)
        # Initialize HybridCipher instances for Alice and Bob.
        self.alice_hybrid = HybridCipher(self.alice_knapsack, self.alice_grid, self.grid_cipher)
        self.bob_hybrid = HybridCipher(self.bob_knapsack, self.bob_grid, self.grid_cipher)

    def run(self):
        # --- Alice sends a message to Bob ---
        alice_message = "Hello, Bob!"
        print("Alice sends:", alice_message)
        ciphertext = self.alice_hybrid.encrypt(alice_message, self.bob_knapsack.public_key)
        print("Ciphertext (Alice->Bob):", ciphertext)
        decrypted_by_bob = self.bob_hybrid.decrypt(ciphertext)
        print("Bob receives:", decrypted_by_bob)

        # --- Bob sends a message to Alice ---
        bob_message = "Hello, Alice!"
        print("\nBob sends:", bob_message)
        ciphertext2 = self.bob_hybrid.encrypt(bob_message, self.alice_knapsack.public_key)
        print("Ciphertext (Bob->Alice):", ciphertext2)
        decrypted_by_alice = self.alice_hybrid.decrypt(ciphertext2)
        print("Alice receives:", decrypted_by_alice)

###############################
# Main Execution
###############################
if __name__ == "__main__":
    protocol = TwoWayHybridProtocol(bit_length=32, toggle_eve=False)
    protocol.run()
