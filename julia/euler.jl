## Helper functions

function ispalindrome(n)
    string(n) == reverse(string(n))
end


function nth_prime_ulimit(n)
    if n < 6
        throw(DomainError())
    end
    return iceil(n * log(n * log(n)))
end


# sieve of eratosthenes
# find all the primes below n
function esieve(n)
    candidates = vcat([2], [3:2:n])
    i = 1
    while i < length(candidates)
        i += 1
        if candidates[i] == 0
            continue
        end
        v = candidates[i]
        candidates[i+v:v:end] = 0
    end
    return filter(n -> n != 0, candidates)
end


## Problems


# Compute all the multiples of 3 and 5 below `ulimit`
function problem1(ulimit=999)
    valid = i -> (i % 3 == 0) | (i % 5 == 0)
    sum(filter(valid, 1:ulimit))
end


# Sum all fibonnacci numbers below `ulimit`
function problem2(i1=0, i2=1, ulimit=4e6)
    sum = 0
    while i2 < ulimit
        if i2 % 2 == 0
            sum += i2
        end
        i1, i2 = i2, i1 + i2
    end
    return sum
end


# compute the largest prime factor of `n`
function problem3(n=600851475143)
    i = 2
    largest_prime_factor = 1
    while n != 1
        if n % i == 0
            largest_prime_factor = i
            n = div(n, i)
        else
            i += 1
        end
    end
    return largest_prime_factor
end


# compute the largest palindrome made from the product of two 3 digit
# numbers
function problem4()
    range = [x for x in 100:999]
    candidates = reverse(sort(kron(range, range)))
    for c in candidates
        if ispalindrome(c)
            return c
        end
    end
end

# compute the smallest multiple of 1:20
function problem5()
    divisors = [19 18 17 16 15 14 13 12 11]
    i = 0
    done = false
    while ~done
        i += 20
        done = true
        for d in divisors
            if i % d != 0
                done = false
                break
            end
        end
    end
    return i
end

# compute square-of-sum - sum-of-squares of 1:ulimit
function problem6(ulimit=100)
    square_of_sum = sum([x for x in 1:ulimit])^2
    sum_of_squares = sum([x^2 for x in 1:ulimit])
    return square_of_sum - sum_of_squares
end


# compute the 10001st prime number
function problem7(n=10001)
    ulimit = nth_prime_ulimit(n)
    return esieve(ulimit)[n]
end


# return the greatest product of 5 consecutive digits
function problem8()
    dta = [7 3 1 6 7 1 7 6 5 3 1 3 3 0 6 2 4 9 1 9 2 2 5 1 1 9 6 7 4 4 2 6 5 7 4 7 4 2 3 5 5 3 4 9 1 9 4 9 3 4 9 6 9 8 3 5 2 0 3 1 2 7 7 4 5 0 6 3 2 6 2 3 9 5 7 8 3 1 8 0 1 6 9 8 4 8 0 1 8 6 9 4 7 8 8 5 1 8 4 3 8 5 8 6 1 5 6 0 7 8 9 1 1 2 9 4 9 4 9 5 4 5 9 5 0 1 7 3 7 9 5 8 3 3 1 9 5 2 8 5 3 2 0 8 8 0 5 5 1 1 1 2 5 4 0 6 9 8 7 4 7 1 5 8 5 2 3 8 6 3 0 5 0 7 1 5 6 9 3 2 9 0 9 6 3 2 9 5 2 2 7 4 4 3 0 4 3 5 5 7 6 6 8 9 6 6 4 8 9 5 0 4 4 5 2 4 4 5 2 3 1 6 1 7 3 1 8 5 6 4 0 3 0 9 8 7 1 1 1 2 1 7 2 2 3 8 3 1 1 3 6 2 2 2 9 8 9 3 4 2 3 3 8 0 3 0 8 1 3 5 3 3 6 2 7 6 6 1 4 2 8 2 8 0 6 4 4 4 4 8 6 6 4 5 2 3 8 7 4 9 3 0 3 5 8 9 0 7 2 9 6 2 9 0 4 9 1 5 6 0 4 4 0 7 7 2 3 9 0 7 1 3 8 1 0 5 1 5 8 5 9 3 0 7 9 6 0 8 6 6 7 0 1 7 2 4 2 7 1 2 1 8 8 3 9 9 8 7 9 7 9 0 8 7 9 2 2 7 4 9 2 1 9 0 1 6 9 9 7 2 0 8 8 8 0 9 3 7 7 6 6 5 7 2 7 3 3 3 0 0 1 0 5 3 3 6 7 8 8 1 2 2 0 2 3 5 4 2 1 8 0 9 7 5 1 2 5 4 5 4 0 5 9 4 7 5 2 2 4 3 5 2 5 8 4 9 0 7 7 1 1 6 7 0 5 5 6 0 1 3 6 0 4 8 3 9 5 8 6 4 4 6 7 0 6 3 2 4 4 1 5 7 2 2 1 5 5 3 9 7 5 3 6 9 7 8 1 7 9 7 7 8 4 6 1 7 4 0 6 4 9 5 5 1 4 9 2 9 0 8 6 2 5 6 9 3 2 1 9 7 8 4 6 8 6 2 2 4 8 2 8 3 9 7 2 2 4 1 3 7 5 6 5 7 0 5 6 0 5 7 4 9 0 2 6 1 4 0 7 9 7 2 9 6 8 6 5 2 4 1 4 5 3 5 1 0 0 4 7 4 8 2 1 6 6 3 7 0 4 8 4 4 0 3 1 9 9 8 9 0 0 0 8 8 9 5 2 4 3 4 5 0 6 5 8 5 4 1 2 2 7 5 8 8 6 6 6 8 8 1 1 6 4 2 7 1 7 1 4 7 9 9 2 4 4 4 2 9 2 8 2 3 0 8 6 3 4 6 5 6 7 4 8 1 3 9 1 9 1 2 3 1 6 2 8 2 4 5 8 6 1 7 8 6 6 4 5 8 3 5 9 1 2 4 5 6 6 5 2 9 4 7 6 5 4 5 6 8 2 8 4 8 9 1 2 8 8 3 1 4 2 6 0 7 6 9 0 0 4 2 2 4 2 1 9 0 2 2 6 7 1 0 5 5 6 2 6 3 2 1 1 1 1 1 0 9 3 7 0 5 4 4 2 1 7 5 0 6 9 4 1 6 5 8 9 6 0 4 0 8 0 7 1 9 8 4 0 3 8 5 0 9 6 2 4 5 5 4 4 4 3 6 2 9 8 1 2 3 0 9 8 7 8 7 9 9 2 7 2 4 4 2 8 4 9 0 9 1 8 8 8 4 5 8 0 1 5 6 1 6 6 0 9 7 9 1 9 1 3 3 8 7 5 4 9 9 2 0 0 5 2 4 0 6 3 6 8 9 9 1 2 5 6 0 7 1 7 6 0 6 0 5 8 8 6 1 1 6 4 6 7 1 0 9 4 0 5 0 7 7 5 4 1 0 0 2 2 5 6 9 8 3 1 5 5 2 0 0 0 5 5 9 3 5 7 2 9 7 2 5 7 1 6 3 6 2 6 9 5 6 1 8 8 2 6 7 0 4 2 8 2 5 2 4 8 3 6 0 0 8 2 3 2 5 7 5 3 0 4 2 0 7 5 2 9 6 3 4 5 0]
    window_size = 5
    greatest = 0

    for start in 1:(length(dta)+1-window_size)
        stop = start + window_size-1
        product = prod(dta[start:stop])
        if product > greatest
            greatest = product
        end
    end
    return greatest
end

# return the pythagorean triple for which a + b + c == 1000
function problem9()
    for c in 1:1000
        for b in 1:c
            for a in 1:b
                if (a + b + c == 1000) && (a^2 + b^2 == c^2)
                    return a*b*c
                end
            end
        end
    end
end


# find the sum of all primes below n
function problem10(n=2000000)
    return sum(esieve(n))
end



problem_11_dta = [
    08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08;
    49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00;
    81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65;
    52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91;
    22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80;
    24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50;
    32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70;
    67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21;
    24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72;
    21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95;
    78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92;
    16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57;
    86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58;
    19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40;
    04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66;
    88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69;
    04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36;
    20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16;
    20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54;
    01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48;
    ]
