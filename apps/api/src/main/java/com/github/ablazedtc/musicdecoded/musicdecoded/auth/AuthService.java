package com.github.ablazedtc.musicdecoded.musicdecoded.auth;

import com.github.ablazedtc.musicdecoded.musicdecoded.common.ApiException;
import com.github.ablazedtc.musicdecoded.musicdecoded.user.User;
import com.github.ablazedtc.musicdecoded.musicdecoded.user.UserRepository;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthService(UserRepository userRepository, PasswordEncoder passwordEncoder, JwtService jwtService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
    }

    public String register(AuthRequest request) {
        if (userRepository.findByEmail(request.getEmail()).isPresent()) {
            throw new ApiException(HttpStatus.CONFLICT, "conflict", "Email already in use");
        }

        User user = new User();
        user.setEmail(request.getEmail());
        user.setPasswordHash(passwordEncoder.encode(request.getPassword()));

        userRepository.save(user);
        return "User registered successfully";
    }

    public String login(AuthRequest request) {
        // Same error for unknown email and wrong password, so callers can't probe which emails have accounts.
        User user = userRepository.findByEmail(request.getEmail())
                .filter(u -> passwordEncoder.matches(request.getPassword(), u.getPasswordHash()))
                .orElseThrow(() -> new ApiException(HttpStatus.UNAUTHORIZED, "unauthorized", "Invalid email or password"));

        return jwtService.generateToken(user.getEmail());
    }
}
