package com.github.ablazedtc.musicdecoded.musicdecoded.auth;

import com.github.ablazedtc.musicdecoded.musicdecoded.common.ApiException;
import com.github.ablazedtc.musicdecoded.musicdecoded.user.User;
import com.github.ablazedtc.musicdecoded.musicdecoded.user.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class AuthServiceTest {

    private final UserRepository users = mock(UserRepository.class);
    private final PasswordEncoder encoder = new BCryptPasswordEncoder(4);
    private final JwtService jwt = new JwtService("test-secret-that-is-at-least-32-bytes-long", 60_000);
    private final AuthService service = new AuthService(users, encoder, jwt);

    private static AuthRequest request(String email, String password) {
        AuthRequest r = new AuthRequest();
        r.setEmail(email);
        r.setPassword(password);
        return r;
    }

    private User existing(String email, String password) {
        User u = new User();
        u.setEmail(email);
        u.setPasswordHash(encoder.encode(password));
        return u;
    }

    @Test
    void unknownEmailAndWrongPasswordGiveTheSame401() {
        when(users.findByEmail("nobody@example.com")).thenReturn(Optional.empty());
        when(users.findByEmail("a@example.com")).thenReturn(Optional.of(existing("a@example.com", "right-password")));

        ApiException unknown = assertThrows(ApiException.class,
                () -> service.login(request("nobody@example.com", "whatever1")));
        ApiException wrong = assertThrows(ApiException.class,
                () -> service.login(request("a@example.com", "wrong-password")));

        for (ApiException e : new ApiException[]{unknown, wrong}) {
            assertEquals(HttpStatus.UNAUTHORIZED, e.getStatus());
            assertEquals("unauthorized", e.getCode());
        }
        assertEquals(unknown.getMessage(), wrong.getMessage());
    }

    @Test
    void correctPasswordReturnsToken() {
        when(users.findByEmail("a@example.com")).thenReturn(Optional.of(existing("a@example.com", "right-password")));

        String token = service.login(request("a@example.com", "right-password"));

        assertEquals("a@example.com", jwt.extractEmail(token));
    }

    @Test
    void duplicateSignupIs409() {
        when(users.findByEmail(anyString())).thenReturn(Optional.of(existing("a@example.com", "x")));

        ApiException e = assertThrows(ApiException.class, () -> service.register(request("a@example.com", "password1")));

        assertEquals(HttpStatus.CONFLICT, e.getStatus());
        assertEquals("conflict", e.getCode());
        verify(users, never()).save(any());
    }
}
