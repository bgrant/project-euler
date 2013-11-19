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
