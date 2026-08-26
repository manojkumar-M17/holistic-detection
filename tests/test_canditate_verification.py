from modules.candidate_verification import CandidateVerifier


verifier = CandidateVerifier()


print("\n==============================")
print("CANDIDATE VERIFICATION TEST")
print("==============================")


# Test 1: Valid student + one face

result = verifier.verify_candidate(
    student_id="24AIML015",
    face_count=1
)

print("\nTest 1:")
print(result)


# Test 2: No face

result = verifier.verify_candidate(
    student_id="24AIML016",
    face_count=0
)

print("\nTest 2:")
print(result)


# Test 3: Multiple faces

result = verifier.verify_candidate(
    student_id="24AIML017",
    face_count=2
)

print("\nTest 3:")
print(result)


# Test 4: Invalid student ID

result = verifier.verify_candidate(
    student_id="",
    face_count=1
)

print("\nTest 4:")
print(result)


# Verification status

print("\n==============================")
print("VERIFICATION STATUS")
print("==============================")

print(
    "24AIML015:",
    verifier.is_verified("24AIML015")
)

print(
    "24AIML016:",
    verifier.is_verified("24AIML016")
)