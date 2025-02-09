"Done by Daksh"

import random

class BB84:
    def __init__(self, bit_length=16, toggle_eve=False, transmission_noise=0.0, measurement_error=0.0, override=False):
        self.bit_length = bit_length
        self.toggle_eve = toggle_eve
        self.transmission_noise = transmission_noise
        self.measurement_error = measurement_error
        self.override = override

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
            if self.override:
                print("\n[BB84] Key Exchange Failed. Overriding...\n")
                return alice_key, bob_key
            
            print("\n[BB84] Key Exchange Failed. Retrying...\n")
            return "", ""

bb84 = BB84(bit_length=32, toggle_eve=False)
alice_key, bob_key = bb84.simulate()
