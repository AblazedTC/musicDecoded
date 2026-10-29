package com.github.ablazedtc.musicdecoded.musicdecoded.auth;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;

@Service
public class JwtService {

    static final int MIN_SECRET_BYTES = 32; // HS256 needs a key of at least 256 bits

    private final SecretKey signInKey;
    private final long expiration;

    public JwtService(@Value("${jwt.secret:}") String secret, @Value("${jwt.expiration}") long expiration) {
        if (secret == null || secret.isBlank()) {
            throw new IllegalStateException(
                    "JWT_SECRET is not set. Set the JWT_SECRET environment variable, "
                            + "or run with SPRING_PROFILES_ACTIVE=dev for local development.");
        }
        byte[] secretBytes = secret.getBytes(StandardCharsets.UTF_8);
        if (secretBytes.length < MIN_SECRET_BYTES) {
            throw new IllegalStateException(
                    "JWT_SECRET is too short: " + secretBytes.length + " bytes, HS256 needs at least "
                            + MIN_SECRET_BYTES + " bytes.");
        }
        this.signInKey = Keys.hmacShaKeyFor(secretBytes);
        this.expiration = expiration;
    }

    public String generateToken(String email) {
        return Jwts.builder()
                .subject(email)
                .issuedAt(new Date())
                .expiration(new Date(System.currentTimeMillis() + expiration))
                .signWith(signInKey)
                .compact();
    }

    public String extractEmail(String token) {
        return Jwts.parser()
                .verifyWith(signInKey)
                .build()
                .parseSignedClaims(token)
                .getPayload()
                .getSubject();
    }

    public boolean isTokenValid(String token) {
        try {
            extractEmail(token);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
