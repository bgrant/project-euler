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


function ispalindrome(n)
    string(n) == reverse(string(n))
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
