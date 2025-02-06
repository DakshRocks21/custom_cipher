import numpy as np

class RotatingGridCipher:
    def __init__(self, size=4):
        self.size = size

    def generate_grid(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        positions = np.random.choice(range(self.size**2), self.size, replace=False)
        for pos in positions:
            grid[pos // self.size][pos % self.size] = 1
        return grid

    def rotate_grid(self, grid):
        return np.rot90(grid, -1)

    def encrypt(self, plaintext):
        grid = self.generate_grid()
        plaintext = plaintext.ljust(self.size ** 2, 'X')
        ciphertext = ['X'] * (self.size ** 2)
        index = 0

        temp_grid = grid.copy()

        for _ in range(4):
            for row in range(self.size):
                for col in range(self.size):
                    if temp_grid[row, col] == 1:
                        ciphertext[row * self.size + col] = plaintext[index]
                        index += 1
            temp_grid = self.rotate_grid(temp_grid) 

        return ''.join(ciphertext), grid

    def decrypt(self, ciphertext, grid):
        plaintext = ['X'] * (self.size ** 2)
        index = 0

        temp_grid = grid.copy()

        for _ in range(4): 
            for row in range(self.size):
                for col in range(self.size):
                    if temp_grid[row, col] == 1:
                        plaintext[index] = ciphertext[row * self.size + col]
                        index += 1
            temp_grid = self.rotate_grid(temp_grid) 

        return ''.join(plaintext).rstrip('X') 

    def print_grid(self, grid):
        print("\nGrid Layout (1s show character positions):")
        for row in grid:
            print(" ".join(str(x) for x in row))

cipher = RotatingGridCipher()

while True:
    print("\n--- Rotating Grid Cipher ---")
    print("1. Encrypt a message")
    print("2. Decrypt a message")
    print("3. Exit")

    choice = input("\nChoose an option: ")

    if choice == "1":
        plaintext = input("\nEnter plaintext: ")
        encrypted_text, grid = cipher.encrypt(plaintext)
        print(f"\n[Encrypted]: {encrypted_text}")
        cipher.print_grid(grid)
        print("\nSave this grid to decrypt the message later!")

    elif choice == "2":
        ciphertext = input("\nEnter ciphertext to decrypt: ")
        print("\nEnter the 4x4 grid manually (0 for empty, 1 for used positions):")

        grid = []
        for i in range(4):
            row = list(map(int, input(f"Row {i+1}: ").split()))
            grid.append(row)

        grid = np.array(grid)
        decrypted_text = cipher.decrypt(ciphertext, grid)
        print(f"\n[Decrypted]: {decrypted_text}")

    elif choice == "3":
        print("\nExiting program. Goodbye!")
        break

    else:
        print("\nInvalid choice. Try again.")

