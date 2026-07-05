package com.github.ablazedtc.musicdecoded.musicdecoded.auth;

import lombok.Data;

@Data
public class AuthRequest {
    private String email;
    private String password;
}
