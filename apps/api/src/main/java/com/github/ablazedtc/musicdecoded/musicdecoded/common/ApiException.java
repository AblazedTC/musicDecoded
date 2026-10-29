package com.github.ablazedtc.musicdecoded.musicdecoded.common;

import org.springframework.http.HttpStatus;

/** An error that maps to the openapi `Error` schema: an HTTP status plus a code from `Error.code`. */
public class ApiException extends RuntimeException {

    private final HttpStatus status;
    private final String code;

    public ApiException(HttpStatus status, String code, String message) {
        super(message);
        this.status = status;
        this.code = code;
    }

    public HttpStatus getStatus() {
        return status;
    }

    public String getCode() {
        return code;
    }
}
