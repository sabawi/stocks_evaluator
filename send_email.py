import os

def send_email(recipient, subject, body):
    # Construct the command to send the email
    command = 'echo "{body}" | mutt -s "{subject}" {recipient}'.format(
        body=body,
        subject=subject,
        recipient=recipient
    )
    
    # Execute the command
    os.system(command)

# Example usage
recipient_email = "sabawi@gmail.com"
email_subject = "Hello World"
email_body = "This is a test email from Python!"

send_email(recipient_email, email_subject, email_body)

