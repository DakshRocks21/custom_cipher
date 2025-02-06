from Crypto.Cipher import ARC2
import binascii
import hashlib

class RC2Cipher:
    def __init__(self, key):
        self.key = hashlib.md5(key.encode()).digest() 
        self.block_size = ARC2.block_size 

    def pad(self, text):
        pad_length = self.block_size - (len(text) % self.block_size)
        return text + (chr(pad_length) * pad_length)

    def unpad(self, text):
        return text[:-ord(text[-1])]

    def encrypt(self, plaintext):
        cipher = ARC2.new(self.key, ARC2.MODE_ECB)
        padded_text = self.pad(plaintext)
        encrypted_bytes = cipher.encrypt(padded_text.encode())
        return binascii.hexlify(encrypted_bytes).decode() 

    def decrypt(self, ciphertext):
        cipher = ARC2.new(self.key, ARC2.MODE_ECB)
        decrypted_bytes = cipher.decrypt(binascii.unhexlify(ciphertext))
        return self.unpad(decrypted_bytes.decode())

key = input("\n[RC2] Enter encryption key: ")
rc2_cipher = RC2Cipher(key)

plaintext = input("\nEnter plaintext to encrypt: ")
ciphertext = rc2_cipher.encrypt(plaintext)
print(f"Encrypted Text (Hex): {ciphertext}")

decrypted_text = rc2_cipher.decrypt(ciphertext)
print(f"Decrypted Text: {decrypted_text}")
